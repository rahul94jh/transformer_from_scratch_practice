import torch
from torch import Tensor
import torch.nn as nn
import transformer_arch as ta
from transformer_arch import SelfAttention, TransformerBlock

class DecoderBlock(nn.Module):
    def __init__(self, embed_size:int=256, heads:int=8, forward_expansion:int=8, dropout:float=0.1, device:str='cpu'):
        super(DecoderBlock, self).__init__()

        # Initilize variables
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size=embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:Tensor, value:Tensor, key:Tensor, src_mask:None, tgt_mask:None):
        # 1. Get attention block 
        attention = self.attention(x,x,x,tgt_mask)

        # 2. add skip connection
        out = attention + x

        # 3. normalize and add dropout
        query = self.dropout(self.norm(out))

        # 4. add transformer block - the value and key for decoder comes from encoder
        out = self.transformer_block(value, key, query, src_mask)
    
        return out

class Decoder(nn.Module):
    def __init__(
        self, 
        tgt_vocab_size,
        embed_size,
        heads,
        num_layers,
        max_length,
        dropout, 
        forward_expansion,
        device,
        ):

        super(Decoder, self).__init__()

        self.word_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.device = device
        
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, enc_out, src_mask, tgt_mask):
        # get batch size and sequence length
        N, seq_length = x.shape

        # create positions and expand to batch size
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        # add positional embedding to wordembedding of sequence
        x = self.word_embedding(x) + self.position_embedding(positions)
        # add dropout layer
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, tgt_mask)
        
        out = self.fc_out(x)

        return out



  















    

