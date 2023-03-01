from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
import torch
import pickle
from re import sub
import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm
import torch.nn as nn
import time
import torch.nn.functional as F
import random
import wandb

with open('datasets/en_tokenized.pkl', 'rb') as f:
    en_tokens = pickle.load(f)

with open('datasets/de_tokenized.pkl', 'rb') as f:
    de_tokens = pickle.load(f)

print(len(en_tokens))
assert(len(de_tokens) == len(en_tokens))


tokenizer_de = Tokenizer.from_file("tokenizer_de_25000_start_token_SOS.json")
tokenizer_en = Tokenizer.from_file("tokenizer_en_25000_start_token_SOS.json")


class MultiHeadAttention(nn.Module):
    def __init__(self, h, edim):
        super().__init__()

        self.h = h
        self.edim = edim
        self.dk = self.edim//self.h
        self.key = nn.Linear(self.edim,self.edim)
        self.query = nn.Linear(self.edim,self.edim)
        self.value = nn.Linear(self.edim,self.edim)
        self.linear = nn.Linear(self.edim,self.edim)
        

    def forward(self, key, value,query, mask = None):

        bs = key.shape[0]
        nwords_key = key.shape[1]
        nwords_query = query.shape[1]

        k = self.key(key).reshape(bs, nwords_key, self.h, self.dk).transpose(1,2)
        q = self.query(query).reshape(bs, nwords_query, self.h, self.dk).transpose(1,2)
        v = self.value(value).reshape(bs, nwords_key, self.h, self.dk).transpose(1,2)
        x = torch.einsum('bhmd,bhnd -> bhmn',(q,k))
        
        if mask != None:
            x = x.masked_fill(mask == False, float("-1e10"))

        x = F.softmax(x/(self.dk)**0.5, dim=3)

        x = torch.einsum('bhmn,bhnv -> bhmv', (x,v))
        x = x.transpose(1,2)

        x = x.reshape(bs, nwords_query, -1)
        x = self.linear(x)
        return x
    

class EncoderBlock(nn.Module):
    def __init__(self, edim, h, hdim, dropout):
        super().__init__()

        self.multiHeadAttention = MultiHeadAttention(h, edim)
        self.norm1 = nn.LayerNorm(edim)
        self.norm2 = nn.LayerNorm(edim)
        self.fc1 = nn.Linear(edim, hdim)
        self.fc2 = nn.Linear(hdim, edim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, src_embed, src_mask):

        x = self.multiHeadAttention(src_embed,src_embed,src_embed, src_mask)
        x = self.dropout1(x)
        subLayer1 = self.norm1(x + src_embed)

        x = self.fc2(self.relu(self.fc1(subLayer1)))
        x = self.dropout2(x)
        subLayer2 = self.norm2(x + subLayer1)

        return subLayer2

class DecoderBlock(nn.Module):
    def __init__(self,edim, h, hdim, dropout):
        super().__init__()

        self.multiHeadAttention = MultiHeadAttention(h, edim)
        self.maskedMultiHeadAttention = MultiHeadAttention(h, edim)
        self.norm1 = nn.LayerNorm(edim)
        self.norm2 = nn.LayerNorm(edim)
        self.norm3 = nn.LayerNorm(edim)
        self.fc1 = nn.Linear(edim, hdim)
        self.fc2 = nn.Linear(hdim, edim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt_embed, src_encoded, src_mask, tgt_mask):

        x = self.maskedMultiHeadAttention(tgt_embed, tgt_embed, tgt_embed, tgt_mask)
        x = self.dropout1(x)
        subLayer1 = self.norm1(x + tgt_embed)

        x = self.multiHeadAttention(src_encoded, src_encoded, subLayer1, src_mask)
        x = self.dropout2(x)
        subLayer2 = self.norm2(x + subLayer1)

        x = self.fc2(self.relu(self.fc1(subLayer2)))
        x = self.dropout3(x)
        subLayer3 = self.norm3(x + subLayer2)
        
        return subLayer3


class Encoder(nn.Module):
    def __init__(self, nx, edim, h, hdim,dropout):
        super().__init__()

        self.nx = nx
        self.transformers = nn.ModuleList([EncoderBlock(edim, h, hdim,dropout) for _ in range(nx)])

    def forward(self, src_embed, src_mask):
        for block in self.transformers:
            embed = block(src_embed, src_mask)
        return embed

class Decoder(nn.Module):
    def __init__(self, nx, edim, h, hdim,dropout):
        super().__init__()

        self.nx = nx
        self.transformers = nn.ModuleList([DecoderBlock(edim, h, hdim,dropout) for _ in range(nx)])

    def forward(self, encoded, tgt_embed, src_mask, tgt_mask):

        for block in self.transformers:
            embed = block(tgt_embed, encoded, src_mask, tgt_mask)
        return embed


class CustomTransformer(nn.Module):
    def __init__(self, nx, edim, h, hdim, dropout, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.src_embedding = nn.Embedding(src_vocab_size,edim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size,edim)
        self.encoder = Encoder(nx, edim, h, hdim,dropout)
        self.decoder = Decoder(nx, edim, h, hdim,dropout)
        self.fc = nn.Linear(edim, tgt_vocab_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src_tokens, tgt_tokens, src_mask, tgt_mask, src_pos_embed, tgt_pos_embed, encoded = None):
        
        if encoded == None:

            src_embed = self.src_embedding(src_tokens) + src_pos_embed
            src_embed = self.dropout1(src_embed)
            encoded = self.encoder(src_embed, src_mask)

        tgt_embed = self.tgt_embedding(tgt_tokens) + tgt_pos_embed
        tgt_embed = self.dropout2(tgt_embed)
        
        decoded = self.decoder(encoded, tgt_embed, src_mask, tgt_mask)
        output = self.fc(decoded)

        return output, encoded


def loader(en_tokens, de_tokens, bs, src_pad_idx, tgt_pad_idx, device, shuffle = True):
    num_batches = len(en_tokens)//bs
    idxs = [i for i in range(num_batches)]

    if shuffle:
        random.shuffle(idxs)

    for i in idxs:
        max_len = len(en_tokens[(i+1)*bs - 1])
        en_tensor = torch.tensor([enc.tolist() + 
        [src_pad_idx]*(max_len - len(enc)) for enc in en_tokens[i*bs:(i+1)*bs]], device=device)
        
        en_pad = torch.tensor((en_tensor != src_pad_idx),device=device).unsqueeze(1).unsqueeze(2)

        tgt_lens = [len(enc) for enc in de_tokens[i*bs:(i+1)*bs]]
        max_len= max(tgt_lens)
        tgt = torch.tensor([enc.tolist() + 
        [tgt_pad_idx]*(max_len - len(enc)) for enc in de_tokens[i*bs:(i+1)*bs]], device=device)
        
        de_mask = torch.ones(bs, max_len, max_len, device=device).tril().unsqueeze(1)

        labels = torch.cat([tgt[ii][1:tgt_lens[ii]] for ii in range(tgt.shape[0])], dim = 0).to(device)

        yield en_tensor, en_pad, tgt, de_mask, tgt_lens, labels


def positionEmbeding(edim, max_nwords):
    pos_emb = torch.zeros((max_nwords, edim))
    for pos in range(max_nwords):
        for i in range(edim//2):
            pos_emb[pos, 2*i] = np.sin(pos/(10000**(2*i/edim)))
            pos_emb[pos, 2*i + 1] = np.cos(pos/(10000**(2*i/edim)))

    return pos_emb

def remove_large_Sentences(en_tokens, de_tokens, max_size):
    li = []
    for i in range(len(en_tokens)):
        if(len(en_tokens[i]) <= max_size and len(de_tokens[i]) <= max_size):
            li.append(i)

    return li


class NoamOpt:

    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self):
        return self.model_size ** (-0.5) *min(self._step ** (-0.5),  self._step * self.warmup ** (-1.5))

def save_model(PATH):
    state = {
      'state_dict': net.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler_step': scheduler._step
      }
    torch.save(state, PATH)


def translate(sentence, net, device):

    net.eval()
    posEmb = positionEmbeding(edim, max_nwords).to(device)

    src_enc = tokenizer_en.encode(sentence).ids
    src = torch.tensor(src_enc, device=device).unsqueeze(0)
    src_mask = None

    tgt_enc = [0]
    tgt = torch.tensor(tgt_enc, device=device).unsqueeze(0)
    tgt_mask =  torch.ones(1, len(tgt), len(tgt),device=device).tril().unsqueeze(1)

    src_pos_embed = posEmb[:src.shape[1]].unsqueeze(0)
    tgt_pos_embed = posEmb[:tgt.shape[1]].unsqueeze(0)

    output, encoded = net(src, tgt, src_mask, tgt_mask, src_pos_embed, tgt_pos_embed)

    _, predicted_idx = torch.max(output.data[0], 1)
    print(tgt)
    print(predicted_idx.tolist())
    print(tokenizer_de.decode(predicted_idx.tolist()))


    idx = 0
    while(predicted_idx[-1].tolist() != 1):
        
        tgt = torch.cat((tgt, predicted_idx[-1].unsqueeze(0).unsqueeze(0)), 1)
        tgt_mask =  torch.ones(1, tgt.shape[-1], tgt.shape[-1]).tril().unsqueeze(1).to(device)
        output, _ = net(src, tgt, src_mask, tgt_mask, src_pos_embed, tgt_pos_embed, encoded)
        _, predicted_idx = torch.max(output.data[0], 1)
        print(predicted_idx.tolist())
        print(tokenizer_de.decode(predicted_idx.tolist()))
        idx+=1

        if(idx == max_nwords):
            break


def validate(val_en_tokens, val_de_tokens, bs):

    net.val()
    correct, total = 0, 0
    p_bar=tqdm(total=len(val_en_tokens)//bs)

    for src, src_mask, tgt, tgt_mask, tgt_lens, labels in loader(val_en_tokens, val_de_tokens, bs, src_pad_idx, tgt_pad_idx, device):
        p_bar.update(1)

        src_pos_embed = posEmb[:src.shape[1]].unsqueeze(0)
        tgt_pos_embed = posEmb[:tgt.shape[1]].unsqueeze(0)

        outputs, _ = net(src, tgt, src_mask, tgt_mask, src_pos_embed, tgt_pos_embed)

        li = [outputs[ii][:tgt_lens[ii]-1] for ii in range(outputs.shape[0])]
        probs = torch.cat(li, dim = 0)
        predicted = torch.max(probs, 1)[1]
        
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        return correct, total



nx = 6
edim = 512
hdim = 2048
h = 8
src_vocab_size = 25000
tgt_vocab_size = 25000
n_epochs = 10
bs = 32
src_pad_idx = 0
tgt_pad_idx = 2
dropout = 0.1
max_nwords = 100
n_epochs = 100
val_steps = 10000
val_accu = 0
logging = False
bs = 32
step = 0


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
net = CustomTransformer(nx, edim, h, hdim, dropout, src_vocab_size, tgt_vocab_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), betas=[0.9, 0.98])
scheduler = NoamOpt(edim,2500,optimizer)

PATH = "saved_models/n_steps_" + str(step)
state = torch.load(PATH)
net.load_state_dict(state['state_dict'])
optimizer.load_state_dict(state['optimizer'])
scheduler._step = state['scheduler_step']


max_idxs = remove_large_Sentences(en_tokens, de_tokens, max_nwords)
en_tokens = [en_tokens[i] for i in max_idxs]
de_tokens = [de_tokens[i] for i in max_idxs]
assert(len(de_tokens) == len(en_tokens))
posEmb = positionEmbeding(edim, max_nwords).to(device)

if logging:
    wandb.init(project="transformers")
    wandb.config = {
    "nsteps": 500000,
    "batch_size": bs
    }

net.train()
for epoch in range(n_epochs):  # loop over the dataset multiple times

    correct, total = 0, 0
    p_bar=tqdm(total=len(en_tokens)//bs)
  
    for src, src_mask, tgt, tgt_mask, tgt_lens, labels in loader(en_tokens, de_tokens, bs, src_pad_idx, tgt_pad_idx, device):
        p_bar.update(1)

        step+=1
        src_pos_embed = posEmb[:src.shape[1]].unsqueeze(0)
        tgt_pos_embed = posEmb[:tgt.shape[1]].unsqueeze(0)

        optimizer.zero_grad()
        # with torch.cuda.amp.autocast():
        outputs, _ = net(src, tgt, src_mask, tgt_mask, src_pos_embed, tgt_pos_embed)

        li = [outputs[ii][:tgt_lens[ii]-1] for ii in range(outputs.shape[0])]
        probs = torch.cat(li, dim = 0)
        loss = criterion(probs, labels)   

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        loss.backward()
        optimizer.step()
        predicted = torch.max(probs, 1)[1]
        
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        if step%2:
            scheduler.step()

        if logging:
            wandb.log(
                {"loss": loss.data,
                "lr": scheduler._rate,
                "accuracy": correct/(total+1),
                "validation accuracy": val_accu,
                })
        
        if(step%5000 == 0):
            PATH = "saved_models/n_steps_" + str(step)
            save_model(PATH)

        
        if step % val_steps == 0: 
            val_accu = validate(val_en_tokens, val_de_tokens, bs)
            print("step {0} | loss: {1:.4f} | Val Accuracy: {2:.3f} %".format(epoch, loss, val_accu))

            if val_accu > best_accu:
              best_accu = val_accu 
              torch.save(net.state_dict(), 'net_val.pth')
              print("Saving")

            net.train()



print('Finished Training')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
net = CustomTransformer(nx, edim, h, hdim, dropout, src_vocab_size, tgt_vocab_size).to(device)

step = 65000
PATH = "saved_models/n_steps_" + str(step)
state = torch.load(PATH)
net.load_state_dict(state['state_dict'])

sentence = "Please don't do this"
translate(sentence, net, device)