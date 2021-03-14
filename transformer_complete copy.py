import torch
from torch import Tensor
import torch.nn as nn
import transformer_arch as ta
import transformer_encoder as Encoder
import transformer_decoder as Decoder


class Transformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        embed_size=256,
        num_layers=6,
        heads=8,
        dropout=0.1,
        max_length=100,
        forward_expansion=8,
        device='cpu',
        ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            max_length,
            forward_expansion,
            dropout,
            device,
        )

        self.decoder = Decoder(
            tgt_vocab_size,
            embed_size,
            heads,
            num_layers,
            max_length,
            dropout,
            forward_expansion,
            device
        )
         
        seld.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out 

