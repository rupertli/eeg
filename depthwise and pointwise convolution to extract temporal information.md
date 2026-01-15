The `temporal_pointwise` layer works together with `temporal_depthwise` as a **separable convolution** pattern.

## The Two-Step Process:

### Step 1: Depthwise Temporal Conv (extracts temporal patterns)
```python
self.temporal_depthwise = nn.Conv2d(
    F2, F2,
    kernel_size=(1, 16),  # 16-wide kernel along TIME axis
    groups=F2,            # Each channel processed independently
)
```
- Applies a **separate 1D filter** to each of the F2 channels
- Each filter learns temporal patterns (rising, falling, oscillation)
- **No cross-channel mixing** — channels are isolated

### Step 2: Pointwise Conv (mixes channel information)
```python
self.temporal_pointwise = nn.Conv2d(
    F2, F2,
    kernel_size=(1, 1),   # No spatial/temporal extent
)
```
- **Mixes information across channels** that were processed separately
- Learns which temporal patterns from different channels should combine

## Why not just use a regular conv?

| Approach | Parameters | Effect |
|----------|-----------|--------|
| Regular Conv2d(F2, F2, (1,16)) | F2 × F2 × 16 = **16,384** | Full mixing |
| Depthwise + Pointwise | F2×16 + F2×F2 = **1,536** | Same expressivity, 10× fewer params |

## Visual Example:

```
After band_fusion: (B, 32, 1, T)  — 32 channels over time

Depthwise (kernel=16):
  Channel 0: [...] → learns "rising pattern detector"
  Channel 1: [...] → learns "plateau detector"
  ...
  Channel 31: [...] → learns "oscillation detector"
  
Pointwise (1×1):
  Output 0 = 0.5×(rising) + 0.3×(plateau) - 0.2×(oscillation)
  → Combines: "rising then plateau but not oscillating" = specific emotion signature
```

**Summary**: Depthwise extracts temporal features per channel; pointwise combines them efficiently. Together they form a separable convolution — same power as regular conv, far fewer parameters.
