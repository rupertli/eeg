 Here are the main approaches:

## 1. Adaptive Pooling (Simplest)

PyTorch's built-in `AdaptiveAvgPool1d` / `AdaptiveMaxPool1d` automatically handles variable input → fixed output:

```python
# Input: (B, C, T_variable) → Output: (B, C, N_fixed)
self.pool = nn.AdaptiveAvgPool1d(output_size=8)  # Always outputs 8 time bins
```

---

## 2. Temporal Pyramid Pooling (TPP) - **Recommended for EEG**

Splits into multiple scales, captures both local and global patterns:

```python
class TemporalPyramidPooling(nn.Module):
    """Split variable-length sequence into fixed intervals at multiple scales."""
    
    def __init__(self, levels=[1, 2, 4, 8]):
        """
        Args:
            levels: Number of bins at each pyramid level
                    [1, 2, 4, 8] = 1+2+4+8 = 15 features per channel
        """
        super().__init__()
        self.levels = levels
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(level) for level in levels
        ])
        self.max_pools = nn.ModuleList([
            nn.AdaptiveMaxPool1d(level) for level in levels
        ])
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T) - variable T
        Returns:
            out: (B, C * sum(levels) * 2) - fixed size
        """
        batch_size, channels, _ = x.shape
        outputs = []
        
        for avg_pool, max_pool in zip(self.pools, self.max_pools):
            avg_out = avg_pool(x)  # (B, C, level)
            max_out = max_pool(x)  # (B, C, level)
            outputs.append(avg_out)
            outputs.append(max_out)
        
        # Concatenate all levels
        out = torch.cat(outputs, dim=2)  # (B, C, total_bins)
        out = out.view(batch_size, -1)   # (B, C * total_bins)
        
        return out
```

**Output size:** For `levels=[1, 2, 4, 8]` with 51 channels:
- Total bins = (1+2+4+8) × 2 (avg+max) = 30
- Output = 51 × 30 = **1530 features**

---

## 3. Segment & Pool with Statistics

Divide into N segments, compute multiple statistics per segment:

```python
class SegmentPooling(nn.Module):
    """Divide into N equal segments, compute stats per segment."""
    
    def __init__(self, n_segments=4):
        super().__init__()
        self.n_segments = n_segments
        # 4 stats per segment: mean, std, max, min
        self.n_stats = 4
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C) - variable T
        Returns:
            out: (B, n_segments * n_stats, C)
        """
        B, T, C = x.shape
        
        # Split into segments using adaptive pooling trick
        x = x.permute(0, 2, 1)  # (B, C, T)
        
        # Compute segment boundaries
        segment_size = T // self.n_segments
        outputs = []
        
        for i in range(self.n_segments):
            start = i * T // self.n_segments
            end = (i + 1) * T // self.n_segments
            segment = x[:, :, start:end]  # (B, C, seg_len)
            
            # Statistics per segment
            seg_mean = segment.mean(dim=2)  # (B, C)
            seg_std = segment.std(dim=2)    # (B, C)
            seg_max = segment.max(dim=2)[0] # (B, C)
            seg_min = segment.min(dim=2)[0] # (B, C)
            
            outputs.extend([seg_mean, seg_std, seg_max, seg_min])
        
        out = torch.stack(outputs, dim=1)  # (B, n_segments*4, C)
        return out.view(B, -1)  # (B, n_segments * 4 * C)
```

---

## 4. Attention-based Temporal Pooling (Most Powerful)

Learns which time steps are important:

```python
class AttentionTemporalPooling(nn.Module):
    """Learn attention weights over time, then weighted pooling."""
    
    def __init__(self, input_dim, n_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))  # Learnable query
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C) - variable T
        Returns:
            out: (B, C) - single vector summary
        """
        B = x.size(0)
        query = self.query.expand(B, -1, -1)  # (B, 1, C)
        
        # Attention: query attends to all time steps
        out, _ = self.attention(query, x, x)  # (B, 1, C)
        return out.squeeze(1)  # (B, C)
```

---

## Comparison

| Method | Captures Temporal Dynamics | Complexity | Output Size |
|--------|---------------------------|------------|-------------|
| **AdaptivePool** | Basic | Low | Fixed N bins |
| **Temporal Pyramid** | Multi-scale ✅ | Medium | sum(levels) × C |
| **Segment + Stats** | Local statistics | Medium | N_seg × 4 × C |
| **Attention** | Learned importance | High | C |

---
