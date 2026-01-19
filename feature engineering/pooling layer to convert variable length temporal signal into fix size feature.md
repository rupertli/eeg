For SEED data set, the duration of trial is different, pooling layer is used to convert the variable length dimension into fixed dimension. 

To capture more temporal feature, we can use mean, max and variance pooling to get different temporal statics:

- **Avg**: general activation level
- **Max**: peak/strongest response  
- **Var**: temporal dynamics/stability 


| Statistic | What it captures | Emotion relevance |
|-----------|------------------|-------------------|
| **Avg pooling** | Mean activation level | Overall emotional intensity |
| **Max pooling** | Peak response | Strongest emotional moment |
| **Var pooling** | Temporal variability | Emotional stability/fluctuation |

## New feature dimension:
- Before: `F2 = 32` features
- After: `F2 × 3 = 96` features (32 avg + 32 max + 32 var)

## Example interpretation:
- High **avg** + high **var** → sustained but fluctuating emotion (e.g., excitement)
- High **max** + low **var** → brief spike (e.g., startle/fear)
- Low **var** across channels → stable emotional state (e.g., neutral)

This is a common technique in EEG analysis and should help capture more nuanced temporal dynamics.
