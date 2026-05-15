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

A Streamlit web application provides an interactive demonstration of all 8 trained models running on live ECG data.

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

**Actual class distribution (from loaded dataset after single-label assignment):**

| Class | Description | Train N | Train % | Test N | Test % |
|---|---|---|---|---|---|
| NORM | Normal ECG | 7,266 | 42.4% | 1,817 | 42.4% |
| MI | Myocardial Infarction | 2,183 | 12.7% | 546 | 12.7% |
| STTC | ST/T-wave Change | 3,643 | 21.2% | 911 | 21.3% |
| HYP | Hypertrophy | 431 | 2.5% | 107 | 2.5% |
| CD | Conduction Disturbance | 3,621 | 21.1% | 905 | 21.1% |
| **Total** | | **17,144** | | **4,286** | |

HYP is the minority class at 2.5% — the primary class-imbalance challenge in this dataset.

---

## System Architecture — Three Deployment Tiers

The system is configurable for different clinical and deployment environments:

| Tier | Mode | Models | CV Accuracy | Latency | Use Case |
|---|---|---|---|---|---|
| 1 | Efficient Scan | PCA/SAE/MAE + SVM, XGBoost | 42–55% | 12–22 ms | Real-time triage, resource-constrained |
| 2 | Pretrained FT | SAE/MAE Fine-tuned | 64–66% | ~3,000 ms | Balanced accuracy/speed, clinical support |
| 3 | Full Analysis | Raw + BiLSTM, Raw + 1D-CNN | 68–73% | 1,598–4,098 ms | Hospital-grade diagnostic accuracy |

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
- Reconstructs full unmasked signal (no masking, no sparsity constraint)
- Same training settings for fair architectural comparison
- **Output:** 128-dimensional feature vector per ECG

> Note: "SAE" here refers to the standard (non-masked) autoencoder baseline, not a sparse autoencoder with L1 penalty. Both SAE and MAE share the same Transformer encoder — the only difference is the self-supervised training objective.

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
| SAE Fine-tuned | Pretrained Transformer encoder + Dense(64, relu) + Dropout(0.3) + Softmax(5), Adam lr=1e-4 |
| MAE Fine-tuned | Pretrained Transformer encoder + Dense(64, relu) + Dropout(0.3) + Softmax(5), Adam lr=1e-4 |

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

### Final Test Set Performance (4,286 held-out records)

| Model | Test Acc | F1-Macro | Precision (macro) | Recall (macro) |
|---|---|---|---|---|
| PCA + SVM | 42.39% | 0.119 | 0.085 | 0.200 |
| SAE + SVM | 57.40% | 0.350 | 0.417 | 0.366 |
| MAE + SVM | 57.49% | 0.350 | 0.424 | 0.366 |
| MAE + XGBoost | 55.13% | 0.404 | 0.429 | 0.400 |
| SAE Fine-tuned | 62.62% | 0.523 | 0.535 | 0.581 |
| MAE Fine-tuned | 61.41% | 0.512 | 0.535 | 0.575 |
| Raw + BiLSTM | 65.21% | 0.557 | 0.567 | 0.635 |
| Raw + 1D-CNN | **66.05%** | **0.568** | **0.582** | **0.650** |

### Per-Class F1 Score (Final Test Set)

| Model | NORM | MI | STTC | HYP | CD |
|---|---|---|---|---|---|
| PCA + SVM | 0.595 | 0.000 | 0.000 | 0.000 | 0.000 |
| SAE + SVM | 0.723 | 0.058 | 0.494 | 0.000 | 0.473 |
| MAE + SVM | 0.735 | 0.068 | 0.459 | 0.000 | 0.487 |
| MAE + XGBoost | 0.723 | 0.249 | 0.474 | 0.089 | 0.488 |
| SAE Fine-tuned | 0.835 | 0.428 | 0.521 | 0.278 | 0.553 |
| MAE Fine-tuned | 0.824 | 0.421 | 0.504 | 0.265 | 0.546 |
| Raw + BiLSTM | 0.838 | 0.518 | 0.530 | 0.288 | 0.610 |
| Raw + 1D-CNN | **0.845** | **0.539** | **0.532** | **0.292** | **0.633** |

### Key Findings

- **1D-CNN achieves the best overall performance** — 72.65% CV accuracy and 66.05% test accuracy with the fastest DL inference at 1,598 ms
- **PCA is the lower bound** — predicts only the NORM majority class (F1=0 for all other classes), confirming that linear compression of the 12,000-dim raw signal destroys discriminative features
- **SVM cannot detect HYP** — F1=0.00 for all three SVM variants; the 2.5% minority class cannot be separated using fixed 128-dim features with a linear classifier
- **End-to-end training enables minority class detection** — HYP F1 jumps from 0.00 (SVM) to 0.09 (XGBoost) to 0.28–0.29 (fine-tuned DL) to 0.29 (raw DL)
- **MAE outperforms SAE in CV** — +1.44% CV accuracy at fine-tuned tier (65.60% vs 64.16%); however this ordering reverses on the held-out test set (SAE: 62.62%, MAE: 61.41%), a reversal consistent with estimation variance given only 10 DL CV folds
- **Efficiency trade-off** — MAE+SVM at 20.5 ms vs 1D-CNN at 1,598 ms: 78× faster with 18% accuracy cost (CV basis)

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
- Validated on a correctly classified MI sample — 1D-CNN confidence: **99.2%**
- High-attention regions align with clinically relevant ECG features (QRS complexes, ST-segment)

---

## Streamlit Demo Application

An interactive web application (`app.py`) allows live ECG classification using all 8 trained models.

**Features:**
- Two ECG input modes: pre-processed test set (4,286 samples) or raw PTB-XL dataset (21,430 records via `wfdb`)
- Four configurable deployment modes matching the research tiers: ML Baselines, Pre-trained + Fine-tuned, End-to-End DL, Compare All 8
- Real-time inference with per-model confidence, probability bar charts, and majority-vote consensus
- German-to-English translation of PTB-XL doctor's reports (`deep_translator`)
- Full validation metrics: CV results table, F1 heatmap, confusion matrices, ROC curves, precision/recall heatmaps

**To run locally:**

```bash
pip install -r requirements_app.txt
streamlit run app.py
```

Requires `models/` (trained model files) and `results/` (test arrays and figures) — not included in this repository due to file size. See `.gitignore`.

---

## Output Files

| File | Description |
|---|---|
| `cv_results_summary.csv` | Full CV results with 95% confidence intervals — authoritative numbers |
| `dunn_posthoc.csv` | Dunn post-hoc pairwise p-values (Bonferroni corrected) |

### Visualizations

| File | Description |
|---|---|
| `outputs_training_curves.png` | SAE vs MAE loss convergence curves |
| `outputs_mae_masking.png` | Original ECG vs 50%-masked input |
| `outputs_reconstruction.png` | Original / masked / MAE reconstructed (3-panel) |
| `outputs_accuracy_cv.png` | CV mean ± std bar chart — all 8 models, color-coded by tier |
| `outputs_f1_heatmap.png` | Per-class F1 heatmap (8 models × 5 classes) |
| `outputs_latency.png` | Inference latency comparison across models |
| `outputs_tsne.png` | t-SNE latent space — SAE vs MAE side by side |
| `outputs_saliency.png` | XAI gradient saliency on correctly classified MI sample (confidence: 99.2%) |
| `outputs_precision_recall.png` | Per-class precision and recall heatmaps |
| `outputs_roc.png` | AUC-ROC curves for all 8 models |
| `outputs_confusion.png` | Confusion matrices for all 8 models |
| `outputs_posthoc.png` | Dunn post-hoc p-value heatmap |

---

## How to Run

### Streamlit App (local — no GPU required)

```bash
pip install -r requirements_app.txt
streamlit run app.py
```

### Experiment Notebook (Kaggle — GPU recommended)

1. Upload `Configurable ECG Classification System.ipynb` to a new Kaggle notebook
2. Add PTB-XL dataset: search `khyeh0719/ptb-xl-dataset` in Data sources
3. Enable **GPU T4 x1** under Session options
4. Click **Save & Run All (Commit)**
5. Estimated runtime: ~2.5 hours on T4 GPU

### Utility Scripts

| Script | Purpose |
|---|---|
| `fix_ml_models.py` | Retrains the 4 ML classifiers using local encoders and X_train — fixes feature alignment |
| `regen_figures.py` | Regenerates the 4 test-set figures (confusion, F1, precision/recall, ROC) from current saved models |

### Dependencies (experiment notebook)

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
2. **HYP class difficulty** — Best model achieves F1=0.29 for HYP (2.5% of data). Class imbalance persists despite balanced weighting.
3. **Single dataset** — Only PTB-XL evaluated. Generalizability to other ECG datasets not tested.
4. **Small Transformer** — 2 blocks, 4 heads, embed_dim=64. Larger models may further improve MAE advantage but exceeded Kaggle T4 compute budget.
5. **MAE vs SAE not statistically significant** — Consistent numerical superiority of MAE in CV, but Bonferroni-corrected Dunn test shows p=1.0 between MAE Fine-tuned and SAE Fine-tuned. 10 DL folds limits statistical power.
6. **CPU/RAM not profiled** — Only inference latency measured. Full resource profiling was not implemented.

---

## Repository Structure

```
RP4_ECG_CLASSIFICATION/
├── app.py                                        # Streamlit demo application (all 8 models)
├── Configurable ECG Classification System.ipynb  # Full experiment pipeline (cells c01-c25)
├── requirements_app.txt                          # Python dependencies for the Streamlit app
├── fix_ml_models.py                              # Utility: retrain ML classifiers locally
├── regen_figures.py                              # Utility: regenerate 4 test-set figures
├── README.md                                     # This file
├── models/                                       # Trained model files (not in repo — too large)
│   ├── model_bilstm.keras
│   ├── model_cnn.keras
│   ├── model_sae_finetuned.keras
│   ├── model_mae_finetuned.keras
│   ├── model_pca_svm.pkl
│   ├── model_sae_svm.pkl
│   ├── model_mae_svm.pkl
│   └── model_mae_xgboost.pkl
└── results/                                      # Test arrays, CSVs, figures (not in repo — too large)
    ├── X_test.npy / X_train.npy
    ├── y_test.npy / y_train.npy
    ├── X_test_pca.npy / X_test_sae.npy / X_test_mae.npy
    ├── encoder_sae.keras / encoder_mae.keras
    ├── cv_results_summary.csv
    ├── dunn_posthoc.csv
    └── outputs_*.png
```

---

## License

This project is submitted as part of an MSc thesis at Vytautas Magnus University. For academic use only.
