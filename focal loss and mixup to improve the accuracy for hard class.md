## How Mixup and Focal Loss Work

### **Mixup Augmentation**

Mixup creates "virtual" training examples by linearly interpolating between pairs of samples:

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1-\lambda) y_j$$

Where $\lambda \sim \text{Beta}(\alpha, \alpha)$

**Why it helps Happy classification:**
1. **Softens decision boundaries** - Instead of learning hard class boundaries, the model learns smooth transitions between classes
2. **Reduces overconfidence** - Model can't memorize exact examples; forced to learn generalizable features
3. **Creates virtual Happy-Neutral examples** - Helps model learn the subtle differences between similar emotions

**`mixup_alpha` tuning:**
| Alpha | Effect |
|-------|--------|
| 0.1-0.2 | Conservative mixing, samples stay close to originals |
| **0.4** | Balanced (default) - good for most tasks |
| 0.8-1.0 | Aggressive mixing - more regularization |
| 2.0+ | Very aggressive - $\lambda$ clusters around 0.5 |

---

### **Focal Loss**

Standard Cross-Entropy vs Focal Loss:

$$\text{CE} = -\log(p_t)$$
$$\text{Focal} = -(1-p_t)^\gamma \log(p_t)$$

Where $p_t$ is the predicted probability for the true class.

**The key insight:** $(1-p_t)^\gamma$ is a **modulating factor**:
- **Easy examples** (high $p_t$): factor → 0, loss is down-weighted
- **Hard examples** (low $p_t$): factor → 1, loss remains high

| $p_t$ (correct class prob) | $\gamma=0$ (CE) | $\gamma=2$ | $\gamma=5$ |
|----------------------------|-----------------|------------|------------|
| 0.9 (easy) | 0.105 | 0.001 | 0.00001 |
| 0.5 (medium) | 0.693 | 0.173 | 0.022 |
| 0.1 (hard) | 2.303 | 1.867 | 1.208 |

**Why it helps Happy:**
- Happy samples are often misclassified (low $p_t$) → high loss → model focuses on them
- Easy Sad/Disgust samples contribute less to gradient updates

**`focal_gamma` tuning:**
| Gamma | Effect |
|-------|--------|
| 0 | Standard Cross-Entropy |
| 1.0 | Mild focus on hard examples |
| **2.0** | Standard choice (original paper) |
| 3.0-5.0 | Aggressive focus on hard examples |

---

### **Tuning Guidelines**

**Quick tuning strategy:**

```
# Start with defaults
--mixup-alpha 0.4 --focal-gamma 2.0

# If still overfitting (train >> val):
--mixup-alpha 0.6 --focal-gamma 2.0

# If underfitting (train accuracy low):
--mixup-alpha 0.2 --focal-gamma 1.0

# If specific class still weak:
--focal-gamma 3.0  # Focus more on hard examples
```

**Grid search suggestion for your case:**
| mixup_alpha | focal_gamma | When to try |
|-------------|-------------|-------------|
| 0.2 | 2.0 | If current config underfits |
| 0.4 | 2.0 | ✓ Current (63.5% test) |
| 0.4 | 3.0 | If Happy still weak |
| 0.6 | 2.0 | More regularization |
