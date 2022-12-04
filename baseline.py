# %%
import torch

train_data_file = 'data/train.tsv'
valid_data_file = 'data/test.tsv'
cls_vocab_file = 'data/cls_vocab'

with open(cls_vocab_file) as f:
    res = [i.strip().lower() for i in f.readlines() if len(i.strip()) != 0]
cls_vocab = dict(zip(res, range(len(res))))

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

        
train_data = Dataset(train_data_file,cls_vocab)
valid_data = Dataset(valid_data_file,cls_vocab)

# %%
from transformers import BertTokenizer

checkpoint='bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(checkpoint)

tokenizer

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

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=True, collate_fn=collote_fn)

# %%
from torch import nn
from transformers import AutoModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)  
        self.classifier = nn.Linear(768, 48)  

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0]   
        logits = self.classifier(cls_vectors)
        return logits

model = NeuralNetwork().to(device)  

# %%
from tqdm.auto import tqdm

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
    return total_loss

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
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    valid_acc = test_loop(valid_dataloader, model, mode='Valid')
    if valid_acc > best_acc:
        best_acc = valid_acc
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_acc_{(100*valid_acc):0.1f}_model_weights.bin')
print("Done!")

# %%


# %%



