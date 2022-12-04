# %%
import torch
import subprocess, os, re
import tensorflow as tf
def check_commit():
    with os.popen("git status") as f:
        for line in f:
            if "Untracked files:" in line:
                break
            print(line[:-1])
            if re.search("\s*\w+:\s*\w+\.(py|sh|txt)", line):
                raise ValueError("Not commited")
            
# import tensorflow as tf

check_commit()

train_data_file = 'data/train.tsv'
valid_data_file = 'data/test.tsv'
cls_vocab_file = 'data/cls_vocab'

#定义数据集
class Dataset(torch.utils.data.Dataset): 
    def __init__(self,data_file,cls_vocab):  
        dataset=[]   
        with open(data_file, 'r', encoding='gb18030') as f: 
            lines = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
            lines = [i.split('\t') for i in lines]  
            for label, utt, slots in lines:
                utt = utt 
                dataset.append({"intent":int(cls_vocab[label]),   
                                "utt":utt})
        Data = {}
        for idx, line in enumerate(dataset): 
            sample = line
            Data[idx] = sample
        self.data = Data

    def __len__(self):  
        return len(self.data)
    
    def __getitem__(self, idx):  
        return self.data[idx]


# %%
from transformers import BertTokenizer

checkpoint='bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(checkpoint)

# %%
from torch.utils.data import DataLoader

# 定义批处理函数
def collote_fn(batch_samples):   
    batch_text= []
    batch_label = []
    for sample in batch_samples:
        batch_text.append(sample['utt'])
        batch_label.append(int(sample['intent']))
    X = tokenizer(      
        batch_text,   
        padding=True,    
        truncation=True,  
        return_tensors="pt"  
    )
    y = torch.tensor(batch_label)
    return X, y

# %%
from torch import nn
from transformers import AutoModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)  
        self.linear = nn.Linear(768, 768*2) 
        self.act = nn.ReLU() 
        # out = 48
        self.linear2 = nn.Linear(768*2, 768)
        self.classifier = nn.Linear(768, 48)

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0]
        l1 = self.linear(cls_vectors)
        o1 = self.act(l1)
        l2 = self.linear2(o1)
        o2 = self.act(l2)
        logits = self.classifier(o2)
        return logits

# %%
from tqdm.auto import tqdm
def get_log():
    with os.popen("cd /code_lp/baseline && git log") as f:
        matched_ = False
        for line in f.readlines():
            # print(line)
            matched = re.search("(?<=commit\s)\w{5,5}", line)
            if matched_ and matched:
                break
            if not matched_ and matched:
                matched_ = True
                log = line[matched.start(): matched.end()]
                continue
            if re.search("^(Author|Date):", line):
                continue
            if len(line.split()) > 0:
                note=","+"_".join(line.split())
                log+=note
    return log[:20]


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))   
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1)*len(dataloader)
    
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)  
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()  # 梯度下降
        optimizer.step()  # 参数更新
        lr_scheduler.step() #学习率更新

        total_loss += loss.item()  
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')  # 打印
        progress_bar.update(1)
    return total_loss, total_loss/(finish_batch_num + batch)

def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")
    return correct

# %%
from torch import nn
from transformers import AdamW, get_scheduler

def main():

    model = NeuralNetwork().to(device)  

    with open(cls_vocab_file) as f:
        res = [i.strip().lower() for i in f.readlines() if len(i.strip()) != 0]
    cls_vocab = dict(zip(res, range(len(res))))
    train_data = Dataset(train_data_file,cls_vocab)
    valid_data = Dataset(valid_data_file,cls_vocab)
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
    valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
    learning_rate = 1e-5
    epoch_num = 10

    loss_fn = nn.CrossEntropyLoss()  # 交叉熵
    optimizer = AdamW(model.parameters(), lr=learning_rate)  
    lr_scheduler = get_scheduler(  
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epoch_num*len(train_dataloader),
    )

    total_loss = 0.
    best_acc = 0.
    log = get_log()
    tf_writer = tf.summary.create_file_writer(f"./summary/{log}")
    for t in range(epoch_num):
        print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
        total_loss, mean_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
        valid_acc = test_loop(valid_dataloader, model, mode='Valid')
        if valid_acc > best_acc:
            best_acc = valid_acc
            print('saving new weights...\n')
            torch.save(model.state_dict(), f'epoch_{t+1}_valid_acc_{(100*valid_acc):0.1f}_model_weights.bin')
        with tf_writer.as_default():
            tf.summary.scalar("train_loss", mean_loss, step=t)
            tf.summary.scalar("valid_acc", valid_acc, step=t)
    print("Done!")

if __name__ == "__main__":
    main()
    # print(get_log())