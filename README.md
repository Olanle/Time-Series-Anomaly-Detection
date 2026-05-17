# Time Series Anomaly Detection (Bearing Fault Detection)
Training a PyTorch autoencoder on normal bearing vibration data to detect mechanical faults through reconstruction error analysis.

---

## Problem Statement
**What is a Bearing?**
A bearing is a mechanical component found in almost every rotating machine — motors, pumps, conveyor belts, turbines, and fans. Its job is to reduce friction between moving parts and support rotational motion. When a bearing starts to fail it develops small cracks, pits, or wear on its surface that create distinctive vibration patterns detectable by sensors.

By the time you can hear or feel a bearing failure, the damage is often already severe and costly. The goal of this project is to detect that failure weeks before it happens — saving thousands of dollars in unplanned downtime and equipment damage.

**Why Anomaly Detection?**
In a real factory, normal operating data is abundant but fault data is rare and unpredictable. Training a classifier on known fault types would miss any new, unseen fault pattern. An autoencoder trained only on normal data learns what normal looks like — anything that deviates gets flagged, whether it's a known fault type or a completely new one.

---

Dataset
CWRU Bearing Dataset — Case Western Reserve University
File used: **feature_time_48k_2048_load_1.csv**
|---|---|---|
| Class | Samples | Type |
| Normal_1 | 230 | Healthy bearing |
| Ball_007/014/021 | 690 | Ball fault — 3 severities |
| IR_007/014/021 | 690 | Inner race fault — 3 severities |
| OR_007/014/021 | 690 | Outer race fault — 3 severities|
| Total | 2,300 | --- |

9 time-domain features per sample: max, min, mean, standard deviation, RMS, skewness, kurtosis, crest factor, form factor.
**Data split:**

- Training: 184 normal samples (80%)
- Validation: 46 normal samples (20%)
- Test: 2,070 fault samples (all fault classes)

---

## Why Train on Normal Data Only
The autoencoder is trained exclusively on the 230 normal samples. It learns to compress and reconstruct normal vibration patterns accurately. When fed a faulty signal it has never seen, it fails to reconstruct it — producing high reconstruction error. This error is the anomaly score.

This approach works in deployment because normal data is always available from day one. Fault data is rare, unpredictable, and often unknown in advance.

---

Model Architecture - Autoencoder
Input (9 features)
→ Encoder: 9 → 16 → 8 → 4  (with ReLU)
→ Bottleneck: 4 features
→ Decoder: 4 → 8 → 16 → 9  (with ReLU, no activation on final layer)
→ Output (9 reconstructed features)

**Why no activation on the final decoder layer:**
The decoder must reconstruct continuous values ranging from roughly -3 to 3 after scaling. Applying ReLU would clip all negative values to zero — making it impossible to reconstruct negative features. The raw linear output is required so the model can freely output any real number.

---

Training Setup
|---|---|
| Component | Value |
| Loss Function | MSELoss (Mean Squared Error) |
| Optimizer | Adam |
| Learning Rate | 0.0001 | 
| Epochs | 50 |
| Training Data | Normal samples only |
| Device | CUDA |
Loss dropped consistently from 1.0156 to 0.6079 over 50 epochs.

---

## Anomaly Detection Results
**Threshold**: mean normal error + 2 × standard deviation of normal errors
|---|---|
| Data | Mean Reconstruction Error |
| Normal (validation) | 0.6644 | 
| Faulty (all fault classes) | 3,768.1448 |

## Fault Detection Rate: 100%
Every single faulty bearing signal was correctly flagged as an anomaly. The reconstruction error gap between normal and faulty signals is over 5,000x — making even a simple threshold highly effective.

---

## Reconstruction Error Distribution
The key visualisation of this project shows two separate histograms:

- Normal errors — clustered between 0 and 2.2, all below the threshold line
- Fault errors — ranging from 0 to 25,000, all above the threshold

The scale difference alone tells the story — faulty signals produce reconstruction errors thousands of times larger than normal ones.

---

## Key Learnings

Anomaly detection is more powerful than classification for industrial fault detection — it catches unknown fault types that a classifier would miss
Autoencoders learn by reconstruction — training target is the input itself, there are no labels
The final decoder layer must have no activation function — it needs to output any real number to reconstruct scaled features accurately
Fit the scaler only on normal training data — fault data must be transformed separately to avoid leakage
A reconstruction error gap of 5,000x between normal and faulty signals shows the autoencoder has genuinely learned the structure of normal vibration
100% fault detection in 50 epochs demonstrates that simple architectures can solve real industrial problems when the right approach is chosen

---

## Stack
PyTorch scikit-learn pandas matplotlib numpy
---
Part of a 10-project ML engineering curriculum targeting Edge/Embedded ML roles.
