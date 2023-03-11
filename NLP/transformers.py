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
import argparse
import torch.nn.functional as F
import random
import wandb
from customTransformers import CustomTransformer 

filePath = os.path.dirname(os.path.realpath(__file__)) + '/'

# print("Loading data...")
with open(filePath + '../data/en_tokenized.pkl', 'rb') as f:
    en_tokens = pickle.load(f)

with open(filePath + '../data/de_tokenized.pkl', 'rb') as f:
    de_tokens = pickle.load(f)

print("Number of training examples = ", len(en_tokens))
assert(len(de_tokens) == len(en_tokens))

tokenizer_de = Tokenizer.from_file(filePath + "../data/tokenizer_de_25000.json")
tokenizer_en = Tokenizer.from_file(filePath + "../data/tokenizer_en_25000.json")

parser = argparse.ArgumentParser(description='Transformer')
parser.add_argument('--nx', type=int, default=6, help='number of encoder and decoder blocks')
parser.add_argument('--edim', type=int, default=512, help='embedding dimension')
parser.add_argument('--hdim', type=int, default=2048, help='hidden dimension')
parser.add_argument('--h', type=int, default=8, help='number of heads')
parser.add_argument('--src_vocab_size', type=int, default=25000, help='source vocabulary size')
parser.add_argument('--tgt_vocab_size', type=int, default=25000, help='target vocabulary size')
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--src_pad_idx', type=int, default=0, help='source padding index')
parser.add_argument('--tgt_pad_idx', type=int, default=2, help='target padding index')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--max_nwords', type=int, default=100, help='maximum number of words')
parser.add_argument('--val_steps', type=int, default=10000, help='validate after n steps')
parser.add_argument('--logging', type=bool, default=True, help='logging')
parser.add_argument('--save_freq', type=int, default=10000, help='save frequency steps')


args = parser.parse_args()

nx = 6
edim = 512
hdim = 2048
h = 8
src_vocab_size = 25000
tgt_vocab_size = 25000
n_epochs = 10
src_pad_idx = 0
tgt_pad_idx = 2
dropout = 0.1
max_nwords = 100
n_epochs = 100
val_steps = 10000
logging = True
bs = 128
save_freq_steps = 10000


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

step = 0
val_accu = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
net = CustomTransformer(nx, edim, h, hdim, dropout, src_vocab_size, tgt_vocab_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), betas=[0.9, 0.98])
scheduler = NoamOpt(edim,2500,optimizer)

# PATH = "saved_models/n_steps_" + str(step)
# state = torch.load(PATH)
# net.load_state_dict(state['state_dict'])
# optimizer.load_state_dict(state['optimizer'])
# scheduler._step = state['scheduler_step']

max_idxs = remove_large_Sentences(en_tokens, de_tokens, max_nwords)
en_tokens = [en_tokens[i] for i in max_idxs]
de_tokens = [de_tokens[i] for i in max_idxs]
assert(len(de_tokens) == len(en_tokens))
posEmb = positionEmbeding(edim, max_nwords).to(device)

if logging:
    wandb.init(project="transformers", entity='basujindal123')
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
        
        if(step%save_freq_steps == 0):
            PATH = "n_steps_" + str(step)
            save_model(PATH)

        
        # if step % val_steps == 0: 
        #     val_accu = validate(val_en_tokens, val_de_tokens, bs)
        #     print("step {0} | loss: {1:.4f} | Val Accuracy: {2:.3f} %".format(epoch, loss, val_accu))

        #     if val_accu > best_accu:
        #       best_accu = val_accu 
        #       torch.save(net.state_dict(), 'net_val.pth')
        #       print("Saving")

        #     net.train()



print('Finished Training')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# net = CustomTransformer(nx, edim, h, hdim, dropout, src_vocab_size, tgt_vocab_size).to(device)

# step = 65000
# PATH = "saved_models/n_steps_" + str(step)
# state = torch.load(PATH)
# net.load_state_dict(state['state_dict'])

sentence = "Please don't do this"
translate(sentence, net, device)