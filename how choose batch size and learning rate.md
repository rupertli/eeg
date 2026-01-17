## Relationship Between Batch Size, Learning Rate & Training Set Size

### Core Principle: Gradient Noise vs. Convergence Speed

| Factor | Effect |
|--------|--------|
| **Small batch** | Noisy gradients → better generalization, slower per-epoch |
| **Large batch** | Stable gradients → faster per-epoch, risk of sharp minima |
| **Higher LR** | Faster learning, but can overshoot/diverge |
| **Lower LR** | Stable but slow, can get stuck |

---

### 1. Batch Size ↔ Learning Rate (Linear Scaling Rule)

When you increase batch size by factor $k$, increase LR by factor $k$:

$$\text{LR}_{\text{new}} = \text{LR}_{\text{base}} \times \frac{\text{batch}_{\text{new}}}{\text{batch}_{\text{base}}}$$

**Your case:**
| Batch Size | LR (scaled from 5e-4 @ 32) |
|------------|---------------------------|
| 32         | 5e-4                      |
| 64         | 1e-3                      |
| 128        | 2e-3                      |
| 240        | 3.75e-3                   |

---

### 2. Training Set Size ↔ Batch Size

**Key metric: Updates per epoch** = $\frac{\text{Training Size}}{\text{Batch Size}}$

| Training Size | Batch Size | Updates/Epoch | Quality |
|---------------|------------|---------------|---------|
| 480           | 32         | 15            | ✅ Good  |
| 480           | 64         | 7-8           | ✅ OK    |
| 480           | 128        | 3-4           | ⚠️ Marginal |
| 480           | 240        | 2             | ❌ Too few |

**Rule of thumb:** Aim for **10-100 updates per epoch** for stable training.

---

### 3. Training Set Size ↔ Learning Rate

Smaller datasets need:
- **Lower LR** (less data to average over → noisier gradients)
- **More regularization** (dropout, weight decay)
- **Smaller models** (fewer parameters to fit)

| Dataset Size | Recommended LR Range |
|--------------|---------------------|
| < 500        | 1e-4 to 5e-4        |
| 500 - 5000   | 5e-4 to 1e-3        |
| 5000+        | 1e-3 to 1e-2        |

---

### 4. Practical Formula for Your Dataset (480 training samples)

```
Optimal batch size: 16-64
Updates per epoch: 8-30
LR: 5e-4 to 1e-3
```

**Visual Summary:**

```
Training Set: 480 samples
         │
         ▼
┌─────────────────────────────────────────┐
│  Batch=32, LR=5e-4  │  15 updates/epoch │ ← Recommended
│  Batch=64, LR=1e-3  │   8 updates/epoch │ ← Also good
│  Batch=240, LR=4e-3 │   2 updates/epoch │ ← Too few!
└─────────────────────────────────────────┘
```

---

