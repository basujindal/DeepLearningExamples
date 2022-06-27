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


'''
ToDo
- Remove large sentences
'''

def positionEmbeding(edim, max_nwords):
    pos_emb = torch.zeros((max_nwords, edim))
    for pos in range(max_nwords):
        for i in range(edim//2):
            pos_emb[pos, 2*i] = np.sin(pos/(10000**(2*i/edim)))
            pos_emb[pos, 2*i + 1] = np.cos(pos/(10000**(2*i/edim)))

    return pos_emb
    

class InputEmbedding:
    def __init__(self) -> None:
        pass


class KVQ(nn.Module):
    def __init__(self, edim):
        super().__init__()

        self.key = nn.Linear(edim,edim)
        self.query = nn.Linear(edim,edim)
        self.value = nn.Linear(edim,edim)

    def forward(self, embedK, embedV, embedQ):
        key = self.key(embedK)
        value = self.value(embedV)
        query = self.query(embedQ)

        return key,value,query


class MultiHeadAttention(nn.Module):
    def __init__(self, h, edim, ):
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
        nwords = key.shape[1]

        k = self.key(key).reshape(bs, nwords, self.h, self.dk).transpose(1,2)
        q = self.query(query).reshape(bs, nwords, self.h, self.dk).transpose(1,2)
        v = self.value(value).reshape(bs, nwords, self.h, self.dk).transpose(1,2)
        x = torch.einsum('bhmd,bhnd -> bhmn',(q,k))
        
        if mask != None:
            x = x.masked_fill(self.mask == False, float("-1e20"))

        x = F.softmax(x/(self.dk)**0.5, dim=3)

        x = torch.einsum('bhmn,bhnv -> bhmv', (x,v))
        x = x.transpose(1,2)

        x = x.reshape(bs, nwords, -1)
        x = self.linear(x)
        return x
    

class EncoderBlock(nn.Module):
    def __init__(self, edim, h, hdim):
        super().__init__()

        self.kvq = KVQ(edim)
        self.multiHeadAttention = MultiHeadAttention(h, edim)
        self.norm1 = nn.LayerNorm(edim)
        self.norm2 = nn.LayerNorm(edim)
        self.fc1 = nn.Linear(edim, hdim)
        self.fc2 = nn.Linear(hdim, edim)
        self.relu = nn.ReLU()

    def forward(self, src_embed, src_mask):

        key,value,query = self.kvq(src_embed,src_embed,src_embed)
        x = self.multiHeadAttention(key,value,query, src_mask)
        subLayer1 = self.norm1(x + src_embed)
        x = self.fc2(self.relu(self.fc1(subLayer1)))
        subLayer2 = self.norm2(x + subLayer1)

        return subLayer2

class DecoderBlock(nn.Module):
    def __init__(self,edim, h, hdim):
        super().__init__()
        self.kvq = KVQ(edim)
        self.kvq2 = KVQ(edim)
        self.multiHeadAttention = MultiHeadAttention(h, edim)
        self.maskedMultiHeadAttention = MultiHeadAttention(h, edim)
        self.norm1 = nn.LayerNorm(edim)
        self.norm2 = nn.LayerNorm(edim)
        self.norm3 = nn.LayerNorm(edim)
        self.fc1 = nn.Linear(edim, hdim)
        self.fc2 = nn.Linear(hdim, edim)
        self.relu = nn.ReLU()

    def forward(self, embed, encoded, src_mask, tgt_mask):
        key,value,query = self.kvq(embed, embed, embed)
        x = self.maskedMultiHeadAttention(key,value,query, tgt_mask)
        subLayer1 = self.norm1(x + embed)

        key,value,query = self.kvq2(encoded, encoded, subLayer1)
        x = self.multiHeadAttention(key,value,query, src_mask)
        subLayer2 = self.norm1(x + subLayer1)

        x = self.fc2(self.relu(self.fc1(subLayer2)))
        subLayer3 = self.norm2(x + subLayer2)

        return subLayer3


class Encoder(nn.Module):
    def __init__(self, nx, edim, h, hdim):
        super().__init__()

        self.nx = nx
        self.transformers = nn.ModuleList([EncoderBlock(edim, h, hdim) for _ in range(nx)])

    def forward(self, src_embed, src_mask):
        for block in self.transformers:
            embed = block(src_embed, src_mask)
        return embed

class Decoder(nn.Module):
    def __init__(self, nx, edim, h, hdim):
        super().__init__()

        self.nx = nx
        self.transformers = nn.ModuleList([DecoderBlock(edim, h, hdim) for _ in range(nx)])

    def forward(self, tgt_embed, encoded, src_mask, tgt_mask):

        for block in self.transformers:
            embed = block(tgt_embed, encoded, src_mask, tgt_mask)
        return embed


class CustomTransformer(nn.Module):
    def __init__(self, nx, edim, h, hdim, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size,edim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size,edim)
        self.encoder = Encoder(nx, edim, h, hdim)
        self.decoder = Decoder(nx, edim, h, hdim)
        self.fc = nn.Linear(edim, tgt_vocab_size)

    def forward(self, src_tokens, tgt_tokens, src_mask, tgt_mask):

        src_embed = self.src_embedding(src_tokens)
        tgt_embed = self.tgt_embedding(tgt_tokens)
        encoded = self.encoder(src_embed, src_mask)
        output = self.decoder(encoded, tgt_embed, src_embed, src_mask, tgt_mask)
        # probs = F.softmax(self.fc(output), dim = 2)

        return output









        

            









