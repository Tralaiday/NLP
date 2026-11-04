# Reflection

Transformer self-attention allows direct interaction between all positions and
can process a batch of tokens in parallel. Positional encoding is required
because self-attention alone does not know token order. Multi-head attention
provides several learned projections, while the FFN transforms each token
representation after interaction.

The main limitation is cost: the attention matrix grows as `T²`. Increasing
sequence length therefore increases memory and computation quickly. A
pretrained Transformer in LAB08 addresses the need to learn useful language
representations from scratch.
