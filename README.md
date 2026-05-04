# Development of a Configurable ECG Classification System
### A Comparative Study of Deep Learning and Conventional Machine Learning Approaches

**Author:** Mohammed Irshad Kunnam Puthoor  
**Programme:** MSc Applied Informatics  
**University:** Vytautas Magnus University (VMU), Lithuania  
**Supervisor:** Prof. Ausra Saudargiene  
**Year:** 2026

---

## Overview

This repository contains the full implementation of a configurable ECG classification framework built on the PTB-XL dataset. The system evaluates Masked Autoencoders (MAE) against Standard Autoencoders (SAE) and PCA as feature extraction methods, and compares Deep Learning classifiers (1D-CNN, BiLSTM) with conventional Machine Learning models (SVM, XGBoost) across three deployment tiers.

**Core research question:** Can MAE pretraining produce superior ECG feature representations compared to SAE and PCA, and how do these representations perform when used with conventional ML classifiers versus end-to-end deep learning?

---

## Dataset

| Property | Value |
|---|---|
| Dataset | PTB-XL — Large Publicly Available ECG Dataset |
| Version | 1.0.1 (PhysioNet) |
| Total records | 21,430 ECG records (full dataset) |
| Sampling frequency | 100 Hz |
| Signal length | 1,000 timesteps (10 seconds) |
| Channels | 12 leads |
| Classes | 5 diagnostic superclasses |

**Class distribution:**

| Class | Description | Approx. % |
|---|---|---|
| NORM | Normal ECG | 40% |
| MI | Myocardial Infarction | 20% |
| STTC | ST/T-wave Change | 25% |
| HYP | Hypertrophy | 5% |
| CD | Conduction Disturbance | 15% |

---

## System Architecture — Three Deployment Tiers

The system is configurable for different clinical and deployment environments:

| Tier | Mode | Models | Accuracy | Latency | Use Case |
|---|---|---|---|---|---|
| 1 | Efficient Scan | PCA/SAE/MAE + SVM, XGBoost | 42–55% | 12–44 ms | Real-time triage, wearables, resource-constrained |
| 2 | Pretrained FT | SAE/MAE Fine-tuned | 64–67% | ~3,000 ms | Balanced accuracy/speed, clinical support |
| 3 | Full Analysis | Raw + BiLSTM, Raw + 1D-CNN | 70–74% | 1,500–4,100 ms | Hospital-grade diagnostic accuracy |

---

## Preprocessing Pipeline

1. **Bandpass filtering** — Butterworth filter (order 3), passband 0.5–40.0 Hz, zero-phase via `filtfilt`. Removes baseline wander and high-frequency muscle noise.
2. **Train/Test split** — 80/20 stratified split, `random_state=42`
3. **StandardScaler** — per-channel normalization, fitted on training data only (no data leakage)
4. **Data caching** — preprocessed arrays saved as `.npy` files to skip ~90 min reload on restarts

---

## Feature Extraction Methods

### MAE — Masked Autoencoder (Novel Method)

- **Encoder:** Transformer-based (replaces Conv1D to properly exploit masking)
  - Patchification: 1,000 timesteps → 40 patches of 25 timesteps × 12 channels
  - Linear projection: 300-dim patches → 64-dim embeddings
  - Learnable positional embeddings (`_PosEmbed` custom layer)
  - 2× Transformer blocks: Multi-Head Attention (4 heads, key_dim=16) + FFN (128-dim, GELU) + LayerNorm
  - GlobalAveragePooling → Dense(128) latent vector
- **Masking:** 50% of patches randomly zeroed per record
- **Dynamic re-masking:** New random mask generated every training epoch — forces global reconstruction learning
- **Decoder:** Lightweight MLP (discarded after pretraining)
- **Output:** 128-dimensional feature vector per ECG

### SAE — Standard Autoencoder (Baseline)

- Identical Transformer encoder architecture as MAE
- Reconstructs full unmasked signal (no masking)
- Same training settings for fair comparison
- **Output:** 128-dimensional feature vector per ECG

### PCA — Principal Component Analysis (Lower-Bound Baseline)

- `PCA(n_components=128)` on flattened raw signal (12,000 input dims)
- Intentionally suboptimal — demonstrates why learned representations are needed
- **Output:** 128-dimensional feature vector per ECG

---

## Transformer Encoder Architecture

```
Input: (batch, 1000, 12)
  └─ Reshape: (batch, 40, 300)          # 40 patches of 25 timesteps × 12 channels
  └─ Dense(64)                           # Linear patch projection
  └─ _PosEmbed(40, 64)                   # Learnable positional embedding
  └─ [× 2 Transformer Blocks]
      ├─ MultiHeadAttention(heads=4, key_dim=16, dropout=0.1)
      ├─ Residual + LayerNorm(epsilon=1e-6)
      ├─ Dense(128, GELU) → Dense(64)    # Feed-forward network
      └─ Residual + LayerNorm(epsilon=1e-6)
  └─ GlobalAveragePooling1D()
  └─ Dense(128, ReLU, name='latent_features')
Output: (batch, 128)
```

**Why Transformer over Conv1D:** Conv1D reconstructs signals locally — masking provides no benefit since unmasked regions can be reconstructed without attending to other patches. Transformer attention is global — to reconstruct a masked patch the model must attend to all other visible patches, making MAE pretraining architecturally meaningful.

---

## Classification Models

### Conventional ML (Efficient Scan Tier)

| Model | Configuration |
|---|---|
| SVM | `LinearSVC(C=1.0, max_iter=3000, class_weight='balanced')` wrapped in `CalibratedClassifierCV` |
| XGBoost | `n_estimators=100, max_depth=6`, per-fold `sample_weight` for class imbalance |

### Deep Learning — Fine-tuned (Pretrained FT Tier)

| Model | Configuration |
|---|---|
| SAE Fine-tuned | Pretrained Transformer encoder + Dense(64, name=clf_dense) + Dropout(0.3) + Softmax(5), Adam lr=1e-4 |
| MAE Fine-tuned | Pretrained Transformer encoder + Dense(64, name=clf_dense) + Dropout(0.3) + Softmax(5), Adam lr=1e-4 |

All encoder layers unfrozen for full end-to-end fine-tuning.

### Deep Learning — Raw (Full Analysis Tier)

| Model | Configuration |
|---|---|
| BiLSTM | Bidirectional LSTM(64 units) + GlobalAveragePooling + Dropout(0.3) + Softmax(5) |
| 1D-CNN | Conv1D(32,7) → MaxPool → Conv1D(64,5) → MaxPool → Conv1D(128,3) → GAP → Dropout(0.3) → Softmax(5) |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Patch size | 25 timesteps |
| Mask ratio | 50% |
| Autoencoder epochs (max) | 50, early stopping patience=5 |
| Classifier epochs (max) | 30, early stopping patience=4 |
| Batch size | 64 |
| Latent dimension | 128 |
| Embed dimension | 64 |
| Transformer heads | 4 |
| FF dimension | 128 |
| Transformer blocks | 2 |
| Dropout | 0.3 |
| Fine-tune learning rate | 1e-4 |
| Optimizer | Adam |
| Global random seed | 42 |

---

## Cross-Validation Strategy

| Split | Method | Folds | Repeats | Total Measurements |
|---|---|---|---|---|
| ML models | RepeatedStratifiedKFold | 10 | 5 | 50 per model |
| DL models | RepeatedStratifiedKFold | 5 | 2 | 10 per model |

Class weights recomputed per fold from fold training labels. `tf.keras.backend.clear_session()` called before each DL fold to free GPU memory and reset layer name counters.

---

## Results

### Cross-Validation Performance with 95% Confidence Intervals

| Model | Mode | N | Mean Acc | Std | 95% CI Low | 95% CI High | Latency |
|---|---|---|---|---|---|---|---|
| PCA + SVM | Efficient Scan | 50 | 42.38% | 0.02% | 42.38% | 42.39% | 21.8 ms |
| SAE + SVM | Efficient Scan | 50 | 53.79% | 0.79% | 53.57% | 54.02% | 20.7 ms |
| MAE + SVM | Efficient Scan | 50 | 54.67% | 0.76% | 54.45% | 54.88% | 20.5 ms |
| MAE + XGBoost | Efficient Scan | 50 | 53.10% | 0.91% | 52.85% | 53.35% | 12.1 ms |
| SAE Fine-tuned | Pretrained FT | 10 | 64.16% | 1.13% | 63.46% | 64.86% | 3,008 ms |
| MAE Fine-tuned | Pretrained FT | 10 | 65.60% | 0.90% | 65.05% | 66.16% | 2,984 ms |
| Raw + BiLSTM | Full Analysis | 10 | 68.04% | 1.94% | 66.84% | 69.24% | 4,098 ms |
| Raw + 1D-CNN | Full Analysis | 10 | **72.65%** | 1.59% | 71.66% | 73.63% | 1,598 ms |

### Final Test Set Performance (4,286 test records)

| Model | Test Acc | Precision | Recall | Macro F1 |
|---|---|---|---|---|
| PCA + SVM | 42.4% | 0.085 | 0.200 | 0.119 |
| SAE + SVM | 53.9% | 0.384 | 0.330 | 0.315 |
| MAE + SVM | 54.8% | 0.381 | 0.343 | 0.325 |
| MAE + XGBoost | 54.4% | 0.422 | 0.397 | 0.400 |
| SAE Fine-tuned | 66.3% | 0.552 | 0.564 | 0.552 |
| MAE Fine-tuned | 67.3% | 0.562 | 0.574 | 0.563 |
| Raw + BiLSTM | 69.8% | 0.602 | 0.665 | 0.611 |
| Raw + 1D-CNN | **74.2%** | **0.641** | **0.672** | **0.644** |

### Per-Class F1 Score (Final Test Set)

| Model | NORM | MI | STTC | HYP | CD |
|---|---|---|---|---|---|
| PCA + SVM | 0.60 | 0.00 | 0.00 | 0.00 | 0.00 |
| SAE + SVM | 0.70 | 0.06 | 0.39 | 0.00 | 0.42 |
| MAE + SVM | 0.72 | 0.05 | 0.44 | 0.00 | 0.42 |
| MAE + XGBoost | 0.71 | 0.28 | 0.48 | 0.07 | 0.45 |
| SAE Fine-tuned | 0.80 | 0.41 | 0.65 | 0.27 | 0.63 |
| MAE Fine-tuned | 0.82 | 0.42 | 0.65 | 0.29 | 0.64 |
| Raw + BiLSTM | 0.81 | 0.54 | 0.69 | 0.33 | 0.70 |
| Raw + 1D-CNN | **0.85** | **0.56** | **0.71** | **0.37** | **0.73** |

### Key Findings

- **MAE outperforms SAE** in every category: +0.88% CV accuracy (SVM tier), +1.44% CV (Fine-tuned tier), +1.0% test accuracy
- **PCA is the lower bound** — predicts only the NORM majority class, confirming raw signal flattening (12,000 dims) produces unusable features
- **SVM cannot detect HYP** — F1=0.00 for all SVM variants; minority class too sparse for linear classifiers on fixed features
- **Fine-tuned transformers detect all classes** — HYP F1=0.29 (MAE Fine-tuned) vs 0.00 (SVM), end-to-end training with class weights enables minority class detection
- **Efficiency trade-off** — MAE+SVM at 20.5 ms vs Raw+CNN at 1,598 ms: 77× faster with 19.5% accuracy cost

---

## Statistical Significance

### Kruskal-Wallis H-Test

| Metric | Value |
|---|---|
| H-statistic | 202.3376 |
| p-value | < 0.000001 |
| Result | Highly significant — model distributions are not equal |

Non-parametric test — no normality assumption required.

### Dunn Post-hoc Test (Bonferroni corrected) — Key Pairwise Results

| Comparison | p-value | Significant |
|---|---|---|
| PCA + SVM vs all others | < 0.0001 | Yes — PCA definitively worst |
| MAE + SVM vs SAE + SVM | 0.1793 | No — numerically similar features |
| MAE Fine-tuned vs SAE Fine-tuned | 1.0000 | No — trend present, not statistically significant |
| MAE Fine-tuned vs Raw + 1D-CNN | 1.0000 | No — fine-tuned transformer statistically equivalent to CNN |
| SAE + SVM vs MAE Fine-tuned | 0.0028 | Yes — fine-tuning significantly better than fixed features |
| SAE + SVM vs Raw + 1D-CNN | 0.0001 | Yes — CNN significantly better than fixed features |
| MAE + XGBoost vs MAE Fine-tuned | < 0.0001 | Yes — fine-tuning significantly better than XGBoost |

**Limitation:** ML models evaluated over 50 folds, DL over 10. Unequal N may affect cross-tier Dunn comparisons.

---

## Explainability (XAI)

Gradient-based saliency maps implemented using `tf.GradientTape` on the final 1D-CNN:

- Gradients of predicted class score computed with respect to raw ECG input
- Saliency = mean absolute gradient across all 12 channels, normalized to [0,1]
- Validated on a correctly classified MI sample (confidence displayed)
- High-attention regions align with clinically relevant ECG features (QRS complexes, ST-segment)

---

## Output Files

| File | Description |
|---|---|
| `rp4-notebook4.ipynb` | Complete pipeline (25 cells, c01–c25) |
| `cv_results_summary.csv` | Full CV results with 95% confidence intervals |
| `dunn_posthoc.csv` | Dunn post-hoc pairwise p-values (Bonferroni corrected) |

### Visualizations (generated during Kaggle run)

| File | Description |
|---|---|
| `outputs_training_curves.png` | SAE vs MAE loss convergence curves |
| `outputs_mae_masking.png` | Original ECG vs 50%-masked input |
| `outputs_reconstruction.png` | Original / masked / MAE reconstructed (3-panel) |
| `outputs_accuracy_cv.png` | CV mean ± std bar chart — all 8 models, color-coded by tier |
| `outputs_f1_heatmap.png` | Per-class F1 heatmap (8 models × 5 classes) |
| `outputs_latency.png` | Inference latency comparison across models |
| `outputs_tsne.png` | t-SNE latent space — SAE vs MAE side by side |
| `outputs_saliency.png` | XAI gradient saliency on correctly classified MI sample |
| `outputs_precision_recall.png` | Per-class precision and recall heatmaps |
| `outputs_roc.png` | AUC-ROC curves for all 8 models |
| `outputs_confusion.png` | Confusion matrices for all 8 models |
| `outputs_posthoc.png` | Dunn post-hoc p-value heatmap |

---

## How to Run

### On Kaggle (recommended — GPU required)

1. Upload `rp4-notebook4.ipynb` to a new Kaggle notebook
2. Add PTB-XL dataset: search `khyeh0719/ptb-xl-dataset` in Data sources
3. Enable **GPU T4 x1** under Session options
4. Click **Save & Run All (Commit)**
5. Estimated runtime: ~2.5 hours on T4 GPU

### Dependencies

```
tensorflow >= 2.12
scikit-learn
xgboost
wfdb
scipy
scikit-posthocs
psutil
matplotlib
seaborn
pandas
numpy
```

---

## Limitations

1. **Unequal CV measurements** — ML: 50 folds, DL: 10 folds. May affect Dunn test accuracy for cross-tier comparisons.
2. **HYP class difficulty** — Best model achieves F1=0.37 for HYP (5% of data). Class imbalance persists despite balanced weighting.
3. **Single dataset** — Only PTB-XL evaluated. Generalizability to other ECG datasets not tested.
4. **Small Transformer** — 2 blocks, 4 heads, embed_dim=64. Larger models may further improve MAE advantage but exceeded Kaggle T4 compute budget.
5. **MAE vs SAE not statistically significant** — Consistent numerical superiority of MAE observed but Bonferroni-corrected Dunn test shows p=1.0 between MAE Fine-tuned and SAE Fine-tuned. 10 DL folds limits statistical power.
6. **CPU/RAM not profiled** — Only inference latency measured. Full resource profiling was not implemented.

---

## Repository Structure

```
RP4_ECG_Classification/
├── rp4-notebook4.ipynb       # Main pipeline notebook (cells c01-c25)
├── cv_results_summary.csv    # Cross-validation results with 95% CI
├── dunn_posthoc.csv          # Dunn post-hoc statistical results
└── README.md                 # This file
```

---

## License

This project is submitted as part of an MSc thesis at Vytautas Magnus University. For academic use only.
