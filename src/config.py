import os

# Paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "ptbxl_database.csv")

# Hyperparameters
MAX_SAMPLES = 1000
LATENT_DIM = 32
EPOCHS_AE = 10
EPOCHS_VAE = 10
EPOCHS_CNN = 10
BATCH_SIZE = 64
SAMPLES_PER_CLASS = 21000
TEST_SIZE = 0.2
VAL_SPLIT = 0.1
RANDOM_STATE = 42

# Labels
LABEL_MAP = {"NORM": 0, "MI": 1}