## Pointwise Convolution for Band Fusion

After the spatial convolution, you have `n_bands × D` = 6 feature maps (with D=2, 3 bands):
- 2 spatial patterns from **gamma** band
- 2 spatial patterns from **beta** band  
- 2 spatial patterns from **alpha** band

These are **independent** — each band was processed separately due to `groups=n_bands`.

### What the 1×1 pointwise conv does:

```python
self.band_fusion = nn.Conv2d(
    n_bands * D, F2,      # 6 → 32
    kernel_size=(1, 1),   # No spatial/temporal mixing, only channel mixing
    bias=False
)
```

It learns **weighted combinations** of band-specific features:

```
Output feature 1 = w1·(gamma_spatial1) + w2·(gamma_spatial2) + 
                   w3·(beta_spatial1)  + w4·(beta_spatial2) +
                   w5·(alpha_spatial1) + w6·(alpha_spatial2)
```

### Why this matters for emotion:

| Emotion Pattern | Cross-Band Signature |
|-----------------|---------------------|
| Arousal (high) | High gamma + high beta in frontal |
| Valence (positive) | Left-right alpha asymmetry + frontal theta |
| Fear | High beta + low alpha |

Without band fusion, the model can only see each band in isolation. With it, the model learns combinations like:

- "High frontal gamma **AND** low occipital alpha" → specific emotion signature
- "Beta-alpha ratio pattern" → arousal indicator
