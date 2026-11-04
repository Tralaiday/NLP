# Calculations

## Positional encoding

For `d_model = 4`:

```text
PE(pos, 0) = sin(pos)
PE(pos, 1) = cos(pos)
PE(pos, 2) = sin(pos / 100)
PE(pos, 3) = cos(pos / 100)
```

The vectors for positions `0`, `1`, and `2` are different, so the model can
distinguish token order.

## Multi-head shapes

With `B = 8`, `T = 100`, `d_model = 128`, and `n_heads = 4`:

```text
d_head = 128 / 4 = 32
Q, K, V       : [8, 4, 100, 32]
attention     : [8, 4, 100, 100]
output        : [8, 100, 128]
```

The attention matrix is `T x T` because every token can attend to every other
token.

## Feed-forward network and residual

The FFN applies two linear layers and a non-linearity independently at each
position. The block then uses:

```text
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

The residual path preserves the input signal and makes optimization easier.
