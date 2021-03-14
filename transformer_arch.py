import torch
from torch import Tensor
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size:int=256, heads:int=8, mask:bool=False):
        super(SelfAttention, self).__init__()

        # assign variables
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads 
        self.mask = mask
        
        # check dimension of head
        assert(
            self.head_dim * self.heads == self.embed_size
        ),"Embedding size needs to be divisible by heads"
        
        # create q, k, v through linear layer
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        # unify
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size) 
    
    def forward(self,values:Tensor, keys:Tensor, query:Tensor, mask:None) -> Tensor:
        N = query.shape[0]   # this is number of training example (batch size)
        
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
         
        # split embedding into self.heads
        query = self.queries(query.reshape(N, query_len, self.heads, self.head_dim))
        values = self.values(values.reshape(N, value_len, self.heads, self.head_dim))
        keys = self.keys(keys.reshape(N, key_len, self.heads, self.head_dim))

        # Now we require to do scaled dot product attention
        # 1. first step is to MatMul of query and key
        energy = torch.einsum("nqhd,nkhd -> nhqk",[query, keys])
        # query shape : [N, q, h, d]
        # key shape : [N, k, h, d]
        # energy shape : [N, h, q, k]
        
        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))  # this is for mask attention in decoder

        # 2. Do the scaling of q*k dividing by sqrt of dimension of query
        attention = energy / (self.embed_size ** 0.5)
        
        # 3. Do the softmax to make the scaore lie netween 0 and 1
        attention = torch.softmax(attention, dim=3)
        # attention shape =   (N, heads, query_len, key_len)

        # 4. MatMul the attention with value
        out = torch.einsum("nhql,nlhd->nqhd",[attention, values])
        # attention shape : (N, h, q, l)  # key and value length to be same as l
        # value shape : (N, l, h, d)
        # out shape : (N, q, h, d)

        # 5. Flatten the last two dimension
        out = out.reshape(
           N, query_len, self.heads * self.head_dim
        )

        # 6. Now we do concatenation at last and pass through the fully connected layer

        out = self.fc_out(out)
       # Linear layer doesn't modify the shape, final shape will be
       # (N, query_len, embed_size)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size:int=256, heads:int=8, dropout:float=0.1, forward_expansion:int=8):
        super(TransformerBlock, self).__init__()
        
        self.attention = SelfAttention(embed_size=embed_size, heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value:Tensor, key:Tensor, query:Tensor, mask:None) -> Tensor:
        # 1. get attention block
        attention = self.attention(value, key, query, mask)
        
        # 2. add skip connection
        x = attention + query

        # 3. normalize and add dropout
        x = self.dropout(self.norm1(x))

        # 4. add to feed forward layer
        forward = self.feed_forward(x)

        # 5. add skip connection
        out = forward + x

        # 6. normalize and add dropout
        out = self.dropout(self.norm2(out))

        return out



