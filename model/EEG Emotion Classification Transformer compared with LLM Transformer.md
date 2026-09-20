Updated todo list

Compared with the original Transformer architecture used for sequence-to-sequence language tasks, `de_transformer.py` is an **encoder-only Transformer adapted for EEG trial classification**.

| Component | Traditional Transformer | This DE Transformer |
|---|---|---|
| Input | Discrete token IDs | Continuous DE feature vectors |
| Input representation | Token embeddings | Linear projection from `channels × 5 bands` |
| Architecture | Encoder and decoder | Encoder only |
| Output task | One prediction per output token | One emotion class per EEG trial |
| Sequence summary | Usually token-wise outputs | Learned `[CLS]` token |
| Position information | Sinusoidal or learned encoding | Sinusoidal positional encoding |
| Attention mask | Padding and causal masks | Padding mask only |
| Sequence length | Often fixed/batched tokens | Variable-length EEG trials padded per batch |
| Domain structure | No EEG-specific structure | Optional frequency-band attention and interaction |
| Classes | Vocabulary tokens | 4 or 5 emotion classes |

**Key differences**

1. **Continuous DE input**

Each time window contains:

$$
D=C\times5
$$

features, where $C$ is the selected EEG channel count and five represents delta, theta, alpha, beta, and gamma bands.

Instead of a token embedding table, the model uses:

```python
self.input_norm = nn.LayerNorm(in_dim)
self.input_proj = nn.Linear(in_dim, d_model)
```

2. **Encoder-only architecture**

It uses `nn.TransformerEncoder` and has no Transformer decoder or encoder-decoder cross-attention. This is appropriate because the goal is classification rather than sequence generation.

3. **Learned classification token**

A learned token is prepended:

```python
cls = self.cls_token.expand(B, -1, -1)
x = torch.cat([cls, x], dim=1)
```

After encoding, only its representation is classified:

```python
x_cls = self.norm(x[:, 0, :])
logits = self.head(x_cls)
```

This follows BERT/ViT-style classification rather than the original Transformer.

4. **EEG frequency-band attention**

The optional `FrequencyBandAttention` computes global statistics over time and channels and learns five weights:

$$
w=\sigma\left(W_2\operatorname{ReLU}(W_1s)\right)
$$

These weights adaptively scale the five EEG bands before Transformer encoding.

5. **Optional cross-band interaction**

`FrequencyBandInteraction` explicitly calculates products between all ten unordered frequency-band pairs. A traditional Transformer receives flattened features and does not explicitly model these physiological band relationships.

6. **Variable-length trial handling**

`collate_pad_mask` pads each batch to its longest trial and creates a Boolean padding mask. The mask prevents padded windows from participating in Transformer attention and EEG band statistics.

7. **Trial-level classification head**

The output head is:

```text
CLS embedding → dropout → linear(128) → ReLU
              → dropout → emotion logits
```

It predicts one label for an entire trial instead of producing an output for every time step.

8. **EEG-specific preprocessing**

The dataset supports:

- channel selection;
- `(T, 310) → (T, 62, 5)` reshaping;
- subject-session z-score normalization;
- optional temporal noise, shifting, and scaling augmentation;
- SEED-V and SEED-IV configurations.

These are outside a traditional Transformer.

9. **Strict LOSO evaluation support**

The training code supports leave-one-subject-out evaluation. The test subject is excluded from optimization, enabling measurement of cross-subject emotion generalization.

In summary, it retains the traditional Transformer’s multi-head self-attention and feed-forward encoder blocks, but replaces language-specific token processing with DE projection, EEG band modules, a CLS classifier, padding-aware variable trial handling, and subject-oriented evaluation.
