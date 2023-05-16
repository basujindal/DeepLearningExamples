import torch
import torch.nn as nn
import torch.nn.functional as F

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
    

# class MultiHeadAttention(nn.Module):
#     def __init__(self, h, edim):
#         super().__init__()

#         self.h = h
#         self.edim = edim
#         self.dk = self.edim//self.h
#         self.key = nn.Linear(self.edim,self.edim)
#         self.query = nn.Linear(self.edim,self.edim)
#         # self.value = nn.Linear(self.edim,self.edim)
#         self.linear = nn.Linear(self.edim,self.edim)
        

#     def forward(self, key, value,query, mask = None):

#         bs = key.shape[0]
#         nwords_key = key.shape[1]
#         nwords_query = query.shape[1]

#         k = self.key(key).reshape(bs, nwords_key, self.h, self.dk).transpose(1,2)
#         q = self.query(query).reshape(bs, nwords_query, self.h, self.dk).transpose(1,2)
#         # v = self.value(value).reshape(bs, nwords_key, self.h, self.dk).transpose(1,2)
#         x = torch.einsum('bhmd,bhnd -> bhmn',(q,k))
        
#         if mask != None:
#             x = x.masked_fill(mask == False, float("-1e10"))

#         x = F.softmax(x/(self.dk)**0.5, dim=3)

#         x = torch.einsum('bhmn,bhnv -> bhmv', (x,k))
#         x = x.transpose(1,2)

#         x = x.reshape(bs, nwords_query, -1)
#         x = self.linear(x)
#         return x


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