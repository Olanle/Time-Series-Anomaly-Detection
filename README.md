# Time Series Anomaly Detection (Bearing Fault Detection)

Training a PyTorch autoencoder on normal bearing vibration data to detect mechanical faults through reconstruction error analysis.

---

## What is a Bearing and Why Does This Matter?

A bearing is a mechanical component found in almost every rotating machine — motors, pumps, conveyor belts, turbines, and fans. Its job is to reduce friction between moving parts and support rotational motion. When a bearing starts to fail it develops small cracks and wear that create distinctive vibration patterns detectable by sensors.

By the time you can hear or feel a bearing failure, the damage is often already severe and costly. This project detects that failure before it becomes catastrophic — saving thousands of dollars in unplanned downtime.

---

## Why Anomaly Detection Instead of Classification?

A fault classifier trained on known fault types would miss any new, unseen fault pattern. An autoencoder trained only on normal data learns one thing deeply — what normal looks like. Anything that deviates, whether a known fault type or a completely new one, gets flagged through high reconstruction error.

In real deployments, normal data is always abundant from day one. Fault data is rare and unpredictable. Training only on normal data is not a limitation — it is the correct approach.

---

## Dataset

**CWRU Bearing Dataset** — Case Western Reserve University

| Class | Samples | Type |
|-------|---------|------|
| Normal | 230 | Healthy bearing |
| Ball faults (3 severities) | 690 | Ball damage |
| Inner race faults (3 severities) | 690 | Inner race damage |
| Outer race faults (3 severities) | 690 | Outer race damage |
| **Total** | **2,300** | Perfectly balanced |

**9 features per sample:** max, min, mean, std, RMS, skewness, kurtosis, crest factor, form factor

**Split:**
- Train: 184 normal samples
- Validation: 46 normal samples
- Test: 2,070 fault samples

---

## Model Architecture

```
Input (9 features)
→ Encoder: 9 → 16 → 8 → 4  (ReLU activations)
→ Bottleneck: 4 features
→ Decoder: 4 → 8 → 16 → 9  (ReLU activations, no activation on final layer)
→ Output (9 reconstructed features)
```

**Why no activation on the final layer:** After StandardScaler, feature values range from -3 to 3. ReLU would clip all negative values to zero — making it impossible to reconstruct negative features. The raw linear output is required.

---

## Training Setup

| Component | Value |
|-----------|-------|
| Loss Function | MSELoss |
| Optimizer | Adam (lr=0.0001) |
| Epochs | 50 |
| Training Data | Normal samples only |
| Device | CUDA |

Loss dropped from 1.0156 to 0.6079 over 50 epochs.

---

## Results

**Threshold:** `mean normal error + 2 × std of normal errors`

| Data | Mean Reconstruction Error |
|------|--------------------------|
| Normal (validation) | 0.6644 |
| Faulty (all fault classes) | 3,768.1448 |

The reconstruction error for faulty signals is over 5,000 times higher than normal signals.

**Per-Class Detection Rate:**

| Fault Class | Total Samples | Detected | Detection Rate |
|-------------|---------------|----------|----------------|
| Ball_007_1 | 230 | 230 | 100% |
| Ball_014_1 | 230 | 230 | 100% |
| Ball_021_1 | 230 | 230 | 100% |
| IR_007_1 | 230 | 230 | 100% |
| IR_014_1 | 230 | 230 | 100% |
| IR_021_1 | 230 | 230 | 100% |
| OR_007_6_1 | 230 | 230 | 100% |
| OR_014_6_1 | 230 | 230 | 100% |
| OR_021_6_1 | 230 | 230 | 100% |
| **Overall** | **2,070** | **2,070** | **100%** |

100% detection across all three fault types — ball, inner race, outer race — and all three severity levels including the smallest fault size (007). The reconstruction error gap is so large that no fault comes close to the threshold.

---

## Key Learnings

- Anomaly detection catches unknown fault types that a classifier would miss entirely
- Autoencoders learn by reconstruction — target is the input itself, no class labels needed
- Final decoder layer must have no activation — needs to output any real number including negatives
- Fit scaler only on normal training data — transform fault data separately to avoid leakage
- A 5,000x reconstruction error gap proves the model learned genuine signal structure
- Always use a different variable name when converting tensors to numpy — overwriting breaks cell re-runs
- Always reset dataframe index before using boolean masks to filter numpy arrays — index misalignment is a silent bug

---

## Next Steps

- Test on raw `.mat` signal files instead of pre-extracted features to practise FFT and signal processing
- Export model to TFLite for deployment on a microcontroller — covered in Project 7
- Implement an adaptive threshold that recalibrates as the machine ages and normal operating conditions drift

---

## Stack
`PyTorch` `scikit-learn` `pandas` `matplotlib` `numpy`

---

*Part of a 10-project ML engineering curriculum targeting Edge/Embedded ML roles.*
