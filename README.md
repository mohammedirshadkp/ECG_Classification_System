# RP4 — Automated ECG Analysis System
**Master Thesis | Mohammed Irshad Kunnam Puthoor | MSc Applied Informatics | VMU | 2026**
**Supervisor: Ausra Saudargiene**

---

## Project Title
Development of a Configurable ECG Classification System: A Comparative Study of Deep Learning and Conventional Machine Learning Approaches

## Goal
A modular, dual-mode ECG classification framework that evaluates Masked Autoencoders (MAE) against benchmark feature extraction methods (PCA, Standard Autoencoders) and compares Deep Learning classifiers (1D-CNN, BiLSTM) with conventional ML models (SVM, XGBoost).

---

## System Modes

| Mode | Pipeline | Best For |
|------|----------|----------|
| **Efficient Scan** | MAE features + SVM / XGBoost | Fast screening, resource-limited environments |
| **Full Analysis** | Raw ECG + BiLSTM / 1D-CNN | High-accuracy diagnosis |

---

## Results (10,000 PTB-XL Records, GPU)

| Model | Mode | Accuracy | Macro-F1 | Latency |
|-------|------|----------|----------|---------|
| PCA + SVM | Efficient Scan | 51.9% | 0.223 | 1753ms |
| SAE + SVM | Efficient Scan | 59.7% | 0.352 | 1580ms |
| **MAE + SVM (Ours)** | **Efficient Scan** | **60.0%** | **0.346** | 1640ms |
| MAE + XGBoost | Efficient Scan | 57.1% | 0.334 | **12.8ms** |
| Raw + BiLSTM | Full Analysis | 72.7% | 0.515 | 2163ms |
| **Raw + 1D-CNN** | **Full Analysis** | **76.8%** | **0.631** | 1176ms |

---

## Dataset
- **PTB-XL** — 21,430 clinical 12-lead ECG recordings
- 5 cardiac superclasses: NORM, MI, STTC, HYP, CD
- Sampling rate: 100Hz | Signal length: 10 seconds (1000 samples)

---

## Core Innovation — Masked Autoencoder (MAE)
- Randomly masks **75% of ECG signal patches** during training
- Forces the model to reconstruct the full signal from only 25% visible data
- Self-supervised — no labels needed for feature learning
- Outperforms PCA (51.9%) and SAE (59.7%) baselines

---

## Project Structure

```
RP4_Automated_ECG_Analysis/
├── src/
│   ├── config.py          # All hyperparameters and paths
│   ├── data_loader.py     # PTB-XL loading, filtering, scaling
│   ├── models.py          # MAE, SAE, BiLSTM, 1D-CNN architectures
│   ├── main.py            # Full pipeline runner
│   ├── visualizer.py      # All 10 result visualizations
│   └── utils.py           # Logging utility
├── kaggle_notebook.ipynb  # Self-contained Kaggle GPU notebook
├── requirements.txt       # Python dependencies
└── README.md
```

---

## How to Run

### Local
```bash
pip install -r requirements.txt
python src/main.py
```

### Kaggle (Recommended — GPU)
1. Upload `kaggle_notebook.ipynb` to Kaggle
2. Add PTB-XL dataset via Add Data
3. Enable GPU accelerator
4. Enable Internet toggle
5. Run All

---

## Tech Stack
- Python 3.12
- TensorFlow 2.19 (GPU)
- scikit-learn | XGBoost | wfdb
- NumPy | Pandas | Matplotlib | Seaborn

---

## Visualizations Generated
- MAE masking strategy
- MAE reconstruction quality
- Accuracy comparison (all models)
- Per-class F1 heatmap
- Inference latency benchmarking
- t-SNE latent space visualization
- XAI gradient saliency maps
- Precision & Recall heatmaps
- AUC-ROC curves
- Confusion matrices
