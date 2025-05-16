import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from torch.optim import Adam

#loading dataset
dataset= load_dataset("tweet_eval", "irony")

tokenizer = BertTokenizer.from_pretrained ('bert-base-uncased')

class SarcasmDataset (torch.utils.data.Dataset):
    def __init__ (self, split):
        self.data=dataset[split]
        self.texts = self.data['text']
        self.labels= self.data['label']
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text=self.texts[idx]
        label=self.labels[idx]
        encoding=tokenizer(text, truncation=True, padding ='max_length', max_length=64, return_tensors='pt')

        return {key: val.squeeze(0) for key,val in encoding.items()}, torch.tensor(label)


train_data=SarcasmDataset('train')
val_data= SarcasmDataset('validation')
train_loader = DataLoader(train_data, batch_size=16 , shuffle =True)
val_loader = DataLoader(val_data, batch_size=32)


class SarcasmClassifier (nn.Module):
    def __init__(self):
        super().__init__()
        self.bert =BertModel.from_pretrained ('bert-base-uncased')
        self.classifier=nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid=nn.Sigmoid()
    def forward(self, input_ids, attention_mask):
        outputs= self.bert(input_ids= input_ids, attention_mask=attention_mask)
        pool_output=outputs.pooler_output
        logits=self.classifier(pool_output)
        return self.sigmoid(logits).squeeze(-1)


model= SarcasmClassifier()


device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=2e-5)









for epoch in range(3): 
    print(f"Current epoch {epoch}")
    
    model.train()
    total_loss=0
    for batch, labels in train_loader:
        input_ids=batch['input_ids' ].to(device)
        attention_mask= batch['attention_mask'].to(device)
        labels=labels.float().to(device)
        optimizer.zero_grad()
        outputs=model(input_ids, attention_mask)
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()

        print(f"train loss: {total_loss/len(train_loader):.4f}")
    

    model.eval()
    total_correct = 0
    total =0
    with torch.no_grad():
        for batch, labels in val_loader:
            input_ids= batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels= labels.to(device)
            outputs= model(input_ids, attention_mask)
            preds=(outputs > 0.5).long()
            total_correct+= (preds==labels).sum().item()
            total+=labels.size(0)

    printf(f"Validation accuracy: {total_correct/total: .4f}")
