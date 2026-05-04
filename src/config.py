import os

# Paths
# Adjust these if your folder structure is different
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "ptbxl_database.csv")
SCP_PATH = os.path.join(DATA_DIR, "scp_statements.csv")

# --- UPDATED DATA PARAMETERS ---
MAX_SAMPLES = 1000      # 10 seconds at 100Hz
CHANNELS = 12           # Full 12-lead clinical ECG
TOTAL_RECORDS_TO_LOAD = 3000

# --- MODEL HYPERPARAMETERS ---
LATENT_DIM = 128        # The size of the "Super Feature" vector
EPOCHS_AE = 20          # Increased epochs for better MAE reconstruction
EPOCHS_DL = 20          # Increased epochs for BiLSTM accuracy
BATCH_SIZE = 32         # Optimized for RTX 3050 (4GB VRAM)
RANDOM_STATE = 42

# --- MAE SPECIFIC ---
PATCH_SIZE = 50         # ECG is split into 20 patches (1000/50)
MASK_RATIO = 0.75       # 75% Masking (The core thesis innovation)

# --- EVALUATION ---
TEST_SIZE = 0.2
VAL_SPLIT = 0.1

# --- LABELS (5 Superclasses) ---
LABEL_MAP = {
    "NORM": 0, 
    "MI": 1, 
    "STTC": 2, 
    "HYP": 3, 
    "CD": 4
}