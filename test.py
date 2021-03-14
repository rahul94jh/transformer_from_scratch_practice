import torch
from transformer_arch import TransformerBlock, SelfAttention
from CTransformer import Clasification_Transformer

from transformer_complete import Transformer

# test the attention block architecture
#attn_model = SelfAttention(embed_size=32, heads=2)
#print(attn_model)

#tf_block = TransformerBlock(embed_size=32, heads=2, forward_expansion=2)
#print(tf_block)

model = Clasification_Transformer()
print(model)

