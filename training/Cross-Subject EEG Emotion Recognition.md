In **EEG-based emotion recognition**, **Cross-Subject** means that **the model is trained on EEG data from some subjects and tested on completely different, unseen subjects**.  
This setting evaluates **how well the emotion recognition model generalizes across different people**, rather than memorizing person-specific patterns.

***

## 1. Why “Cross-Subject” Matters in EEG Emotion Recognition

EEG signals are **highly individual-specific** due to differences in:

*   Brain anatomy and physiology
*   Electrode placement and impedance
*   Baseline emotional reactivity
*   Cognitive styles and mental states

If a model works only when trained and tested on the **same subject**, it is usually **not practical** for real-world deployment.

✅ **Cross-subject performance reflects real usability**

***

## 2. Cross-Subject vs. Other Evaluation Protocols

| Protocol            | Training Data     | Testing Data                | What It Measures                 |
| ------------------- | ----------------- | --------------------------- | -------------------------------- |
| **Within‑Subject**  | One subject       | Same subject                | Personalized performance         |
| **Cross‑Trial**     | Some trials       | Other trials (same subject) | Trial robustness                 |
| ✅ **Cross‑Subject** | Multiple subjects | New subjects                | **Generalization across people** |
| Cross‑Session       | One session       | Another session             | Temporal stability               |

📌 **Cross-subject is the hardest but most realistic** evaluation setting.

***

## 3. Typical Cross-Subject Experimental Setup

### Example (SEED / DEAP):

*   Total subjects: **15**
*   Split:
    *   Train on EEG from **14 subjects**
    *   Test on **1 unseen subject**
*   Repeat for all subjects (Leave-One-Subject-Out, LOSO)

👉 Final performance = **average accuracy across all held-out subjects**

***

## 4. Why Cross-Subject EEG Emotion Recognition Is Challenging

### (1) Distribution Shift

EEG feature distributions differ greatly across subjects:

$$
P(X_{\text{subject A}}) \neq P(X_{\text{subject B}})
$$

Even if emotional labels are the same.

***

### (2) Feature Non-Universality

Some features are **emotion-discriminative only for specific subjects**:

*   DE features
*   PSD power
*   CWT time–frequency maps

***

### (3) Small Sample Problem

Datasets typically have:

*   Few subjects (≈15–32)
*   Few trials per emotion  
    → Overfitting risk is high

***

## 5. What “Good” Cross-Subject Performance Means

A model with strong cross-subject performance implies:

✅ Emotion-related representations are **subject-invariant**  
✅ Learned patterns are **neurophysiologically meaningful**  
✅ Better potential for **real-world BCI / affective computing**

***

## 6. Common Techniques Used for Cross-Subject EEG Emotion Recognition

### 6.1 Feature-Level Strategies

| Method                    | Purpose                      |
| ------------------------- | ---------------------------- |
| Differential Entropy (DE) | Reduce amplitude variability |
| Log-PSD / Relative Power  | Normalize power distribution |
| Riemannian features       | Model covariance geometry    |

***

### 6.2 Normalization Across Subjects

*   Z-score per subject
*   Baseline removal
*   Trial-wise normalization

These reduce inter-subject scale differences.

***

### 6.3 Domain Adaptation / Transfer Learning

*   **DANN** (Domain-Adversarial Neural Networks)
*   MMD / CORAL loss
*   Subspace alignment

Goal:

$$
\text{Emotion features} \perp \text{Subject identity}
$$

***

### 6.4 Model-Level Approaches

| Model                  | Why Used                       |
| ---------------------- | ------------------------------ |
| CNN                    | Local spatial pattern learning |
| Transformer            | Temporal + channel attention   |
| GNN                    | EEG electrode topology         |
| Hybrid CNN + Attention | Robust feature selection       |

***

## 7. Cross-Subject vs. Subject-Dependent Accuracy (Reality Check)

Example observation from literature:

| Setting           | Typical Accuracy |
| ----------------- | ---------------- |
| Subject-dependent | 85–95%           |
| ✅ Cross-subject   | **55–75%**       |
| Chance (3-class)  | 33%              |

📉 Lower accuracy is **expected**, not failure.

***

## 8. How to Report Cross-Subject Results Correctly (Academic Best Practice)

When you say *“cross-subject EEG emotion recognition”*, you should specify:

✅ Dataset (SEED, DEAP, SEED‑IV, etc.)  
✅ Splitting protocol (LOSO or k-fold subject-wise)  
✅ Whether any fine-tuning on test subject is used  
✅ Per-subject normalization or not

Example wording:

> “We evaluate our method under a **leave-one-subject-out cross-subject protocol** on the SEED dataset, where EEG data from one subject are completely excluded during training.”

***

## 9. In One Sentence (TL;DR)
 **Cross-subject EEG emotion recognition evaluates whether a model trained on some people can correctly recognize emotions from completely new individuals, reflecting its real-world generalization ability.**

***

