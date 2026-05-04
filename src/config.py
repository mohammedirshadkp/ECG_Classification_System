import os

# Paths
SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
CSV_PATH     = os.path.join(DATA_DIR, "ptbxl_database.csv")
SCP_PATH     = os.path.join(DATA_DIR, "scp_statements.csv")

# --- DATA PARAMETERS ---
MAX_SAMPLES           = 1000   # 10 seconds @ 100 Hz
CHANNELS              = 12     # Full 12-lead clinical ECG
TOTAL_RECORDS_TO_LOAD = None   # None = load ALL ~21,837 records (full PTB-XL)

# --- MODEL HYPERPARAMETERS ---
LATENT_DIM   = 128     # 128-dim feature vector output from all extractors
EPOCHS_AE    = 30      # Autoencoder epochs (EarlyStopping patience=5)
EPOCHS_DL    = 30      # DL classifier epochs (EarlyStopping patience=4/5)
BATCH_SIZE   = 64      # Optimised for Kaggle GPU (P100/T4)
RANDOM_STATE = 42

# --- MAE SPECIFIC ---
PATCH_SIZE = 50        # ECG split into 20 patches (1000/50)
MASK_RATIO = 0.75      # 75% masking — core MAE innovation

# --- CLASSIFIER HYPERPARAMETERS (explicit) ---
SVM_KERNEL   = 'rbf'   # Radial Basis Function kernel
SVM_C        = 1.0     # SVM regularisation parameter
XGB_N_EST    = 100     # XGBoost number of estimators
XGB_DEPTH    = 6       # XGBoost max tree depth
BILSTM_UNITS = 64      # BiLSTM units per direction (128 total)
DROPOUT      = 0.3     # Dropout rate for DL classifiers

# --- CROSS-VALIDATION ---
CV_FOLDS_ML   = 10     # 10-fold CV for ML models
CV_REPEATS_ML = 5      # × 5 repeats = 50 measurements per ML model
CV_FOLDS_DL   = 5      # 5-fold CV for DL models (GPU compute constraint)
CV_REPEATS_DL = 2      # × 2 repeats = 10 measurements per DL model

# --- EVALUATION ---
TEST_SIZE = 0.2
VAL_SPLIT = 0.1

# --- LABELS (5 Superclasses) ---
LABEL_MAP = {
    "NORM": 0,
    "MI":   1,
    "STTC": 2,
    "HYP":  3,
    "CD":   4
}
CLASS_NAMES = list(LABEL_MAP.keys())