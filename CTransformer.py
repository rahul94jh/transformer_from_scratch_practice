import torch.nn.functional as F
import torch
from torch import Tensor
import torch.nn as nn
import transformer_arch as ta

class Clasification_Transformer(nn.Module):
    def __init__(
        self,
        num_classes=4,
        src_vocab_size=20000,
        embed_size=256,
        num_layers=6,
        heads=8,
        max_length=100,
        forward_expansion=8,
        dropout=0.0,
        device='cpu'
        ):

        super(Clasification_Transformer, self).__init__()

        self.embed_size = embed_size
        self.num_classes = num_classes
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        # stack layers of transformer block
        self.layers = nn.ModuleList(
              [
                  ta.TransformerBlock(
                      embed_size, 
                      heads,
                      dropout=dropout,
                      forward_expansion=forward_expansion,
                  ) 
                  for _ in range(num_layers)
              ]
          )
        
        self.toprobs = nn.Linear(self.embed_size, self.num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x:Tensor, mask:None) -> Tensor:
        N, seq_length = x.shape
        # expand positions to batch size
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device) 

            # add word embedding and positional encoding together
        out = self.dropout(
            self.word_embedding(x) + self.position_embedding(positions)
        )
        
        # pass the encoded input to transformer layers
        # in encoder key, query and value are all same 
        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = out.mean(dim=1)  # average pooling

        out = self.toprobs(out)

        return F.log_softmax(x, dim=1)


