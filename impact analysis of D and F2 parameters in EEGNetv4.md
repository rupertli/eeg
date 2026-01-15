Here's how **D** and **F2** increase the model's learning capacity:

## D (Depth Multiplier for Spatial Convolution)

```python
self.spatial_conv = nn.Conv2d(
    n_bands, n_bands * D,  # 3 → 3×D output channels
    kernel_size=(n_channels, 1),
    groups=n_bands,
)
```

| D value | Spatial filters per band | Total spatial patterns |
|---------|--------------------------|------------------------|
| D=2 | 2 per band | 6 (e.g., left-right, front-back) |
| D=4 | 4 per band | 12 (+ diagonal, regional patterns) |
| D=8 | 8 per band | 24 (finer-grained spatial combinations) |

**Why it helps**: Emotions involve complex spatial patterns. D=2 might only capture basic asymmetry, while D=4 can learn:
- Frontal vs parietal activation
- Left-right asymmetry  
- Central vs peripheral differences
- Specific regional clusters

---

## F2 (Number of Feature Channels)

```python
self.band_fusion = nn.Conv2d(n_bands * D, F2, ...)  # D×3 → F2
self.temporal_pointwise = nn.Conv2d(F2, F2, ...)    # F2 → F2
self.fc1 = nn.Linear(F2 * 3, 64)                    # 3×F2 → 64
```

| F2 value | Band-fused features | Final pooled features (×3 stats) |
|----------|---------------------|----------------------------------|
| F2=32 | 32 | 96 |
| F2=64 | 64 | 192 |
| F2=128 | 128 | 384 |

**Why it helps**: F2 determines how many distinct "emotion-relevant patterns" can be represented after combining spatial and band information. More F2 = more nuanced representations:
- F2=32: Can represent ~32 distinct spatio-spectral patterns
- F2=64: Can represent ~64 patterns (e.g., "high gamma frontal + low alpha occipital")

---

## Summary

| Parameter | What it controls | Underfitting fix |
|-----------|------------------|------------------|
| **D** | Richness of spatial patterns per frequency band | Increase to capture more electrode relationships |
| **F2** | Richness of combined spatio-spectral-temporal features | Increase to represent more complex emotion signatures |

**Trade-off**: Higher D and F2 = more parameters = risk of overfitting on small datasets. Watch the train-val gap!
