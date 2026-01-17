## Balancing Dropout and Weight Decay

Both are regularization techniques that prevent overfitting, but they work differently:

| Technique | How it works | Effect |
|-----------|--------------|--------|
| **Dropout** | Randomly zeros neurons during training | Forces redundant representations |
| **Weight Decay** | Penalizes large weights (L2 regularization) | Keeps weights small & smooth |

---

### Key Principle: They're Partially Redundant

Using both at high values = **over-regularization** → underfitting (your 20% accuracy case)

```
High Dropout + High Weight Decay = Model can't learn
Low Dropout + Low Weight Decay = Overfitting risk
```

---

### Recommended Combinations

| Scenario | Dropout | Weight Decay | Notes |
|----------|---------|--------------|-------|
| **Small dataset (<1000)** | 0.3-0.5 | 1e-4 to 1e-3 | Your case (720 samples) |
| **Medium dataset** | 0.4-0.5 | 1e-4 | Standard setup |
| **Large dataset** | 0.5-0.6 | 1e-5 | Less regularization needed |
| **Heavy dropout** | 0.5-0.6 | 1e-5 or 0 | Pick one as primary |
| **Heavy weight decay** | 0.1-0.3 | 1e-2 to 1e-3 | Pick one as primary |

---

### Rule of Thumb

$$\text{Total Regularization} = \text{Dropout} + \text{Weight Decay (scaled)}$$

**If increasing one, decrease the other:**

```
Dropout 0.5 + WD 1e-4  ≈  Dropout 0.3 + WD 1e-3
```

---

### For Your EEG Dataset (480 training samples)

| Setting | Dropout | Weight Decay | Expected |
|---------|---------|--------------|----------|
| ❌ Too strong | 0.6 | 0.01 | Underfitting (20% acc) |
| ✅ Balanced | 0.4 | 1e-4 | Good learning |
| ✅ Dropout-focused | 0.5 | 1e-5 | Also good |
| ✅ WD-focused | 0.2 | 1e-3 | Also good |

---

### Practical Tuning Strategy

```
1. Start with: dropout=0.3, weight_decay=1e-4
2. If overfitting (train_acc >> val_acc): 
   → Increase dropout to 0.4-0.5
3. If underfitting (both accuracies low):
   → Decrease dropout to 0.2, weight_decay to 1e-5
4. Fine-tune one at a time, not both
```

