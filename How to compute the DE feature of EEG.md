

# ✅ **What is Differential Entropy (DE)?**

For a continuous random variable that is **Gaussian distributed**, the differential entropy is:

$$
DE = \frac{1}{2} \ln(2\pi e \sigma^{2})
$$

So DE is fully determined by the **variance** of the signal within a time window.

In most EEG literature, DE is computed **per channel per frequency band**, e.g.:

*   Delta (1–4 Hz)
*   Theta (4–8 Hz)
*   Alpha (8–13 Hz)
*   Beta (13–30 Hz)
*   Gamma (30–45 Hz)

***

# ✅ **Pipeline to calculate DE from EEG raw signal**

### **Step 1. Bandpass filter into classical frequency bands**

Use FIR/IIR filters or wavelet filter banks.

For each EEG channel:

    x_delta = bandpass(raw, 1–4 Hz)
    x_theta = bandpass(raw, 4–8 Hz)
    ...

***

### **Step 2. Segment EEG into windows**

Most common:

*   1-second window with 50% overlap
*   128 or 256 ms windows for high resolution

Let each window be:

$$
x = \{x_1, x_2, ..., x_N\}
$$

***

### **Step 3. Compute variance in each window**

$$
\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2
$$

This step assumes EEG in a narrow band is approximately Gaussian.

***

### **Step 4. Compute DE value**

$$
DE = \frac{1}{2} \ln(2\pi e \sigma^{2})
$$

Equivalent, up to a constant, to:

$$
DE = \ln(\sigma)
$$

(Which is why DE is often interpreted as the logarithm of band-power.)

***

# 📌 **Python example (Ready‑to‑use)**

```python
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass(data, low, high, fs, order=4):
    b, a = butter(order, [low/fs*2, high/fs*2], btype='band')
    return filtfilt(b, a, data)

def differential_entropy(signal):
    var = np.var(signal, ddof=0)
    return 0.5 * np.log(2 * np.pi * np.e * var)

# Example: compute DE for one channel
def compute_de(raw_signal, fs=128, win_len=1.0):
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, 45),
    }
    
    win_samples = int(win_len * fs)
    result = {}

    for name, (low, high) in bands.items():
        filtered = bandpass(raw_signal, low, high, fs)
        DE_values = []

        for start in range(0, len(filtered) - win_samples, win_samples):
            seg = filtered[start:start + win_samples]
            DE_values.append(differential_entropy(seg))
        
        result[name] = np.array(DE_values)

    return result
```

***

# 🎯 **Interpretation**

DE and logarithmic bandpower are **monotonically related**:

$$
DE \propto \log(\sigma^{2}) = \log(\text{band power})
$$

Therefore:

*   Higher **beta/gamma DE** → higher arousal
*   Higher **alpha DE** → lower mental load / relaxation
*   DE features are widely used in CNN/RNN/Transformer EEG models in affective computing

***

