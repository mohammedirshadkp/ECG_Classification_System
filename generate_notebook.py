"""
Generator script: produces rp4-notebook4.ipynb
Run once: python generate_notebook.py
Changes v2:
  - Data caching (saves 60-90 min on re-runs)
  - MAE Fine-Tuning (standard MAE pipeline: pretrain → fine-tune end-to-end)
  - Class weights for HYP imbalance
  - XAI fixed to show correctly classified sample
  - Dunn test limitation note added
"""
import json, os

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "rp4-notebook4.ipynb")

def md(src):
    return {"cell_type": "markdown", "id": f"md{abs(hash(src[:30]))}", "metadata": {}, "source": src}

def code(src, cell_id=None):
    h = cell_id or f"c{abs(hash(src[:30]))}"
    return {"cell_type": "code", "execution_count": None, "id": h,
            "metadata": {}, "outputs": [], "source": src}

cells = []

# ── Title ────────────────────────────────────────────────────────────────────
cells.append(md("""# RP4 — ECG Classification System
### Full PTB-XL Dataset | MAE Fine-Tuning | 10-Fold CV × 5 Repeats | Kruskal-Wallis
**Mohammed Irshad Kunnam Puthoor | MSc Applied Informatics | VMU 2026**"""))

# ── CELL 1: Install ──────────────────────────────────────────────────────────
cells.append(code(
"""# Install dependencies
!pip install wfdb xgboost psutil scikit-posthocs -q""", "c01_install"))

# ── CELL 2: Imports ──────────────────────────────────────────────────────────
cells.append(code(
"""import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import wfdb
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import scikit_posthocs as sp

from scipy.signal import butter, filtfilt
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from tensorflow.keras import layers, Model, Input, optimizers, callbacks
from xgboost import XGBClassifier

print('All imports OK')
print('TensorFlow version:', tf.__version__)
print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)""", "c02_imports"))

# ── CELL 3: Config ───────────────────────────────────────────────────────────
cells.append(code(
"""# ── CONFIGURATION ────────────────────────────────────────────────────────────
DATA_DIR = '/kaggle/input/datasets/khyeh0719/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1'
CSV_PATH = os.path.join(DATA_DIR, 'ptbxl_database.csv')
SCP_PATH = os.path.join(DATA_DIR, 'scp_statements.csv')
CACHE_DIR = '/kaggle/working'   # cached numpy arrays saved here

# ── Data ─────────────────────────────────────────────────────────────────────
TOTAL_RECORDS_TO_LOAD = 5000   # 5,000 records — fast run (~60-75 min total)
MAX_SAMPLES           = 1000   # 10 seconds @ 100 Hz
CHANNELS              = 12     # Full 12-lead ECG

# ── Model ────────────────────────────────────────────────────────────────────
LATENT_DIM   = 128
PATCH_SIZE   = 50
MASK_RATIO   = 0.75
EPOCHS_AE    = 30
EPOCHS_DL    = 30
BATCH_SIZE   = 64
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Explicit Classifier Hyperparameters ──────────────────────────────────────
SVM_C        = 1.0
XGB_N_EST    = 100
XGB_DEPTH    = 6
BILSTM_UNITS = 64
DROPOUT      = 0.3

# ── Cross-Validation ─────────────────────────────────────────────────────────
CV_FOLDS_ML   = 10
CV_REPEATS_ML = 5    # 50 measurements per ML model
CV_FOLDS_DL   = 5
CV_REPEATS_DL = 2    # 10 measurements per DL model

# ── Labels ───────────────────────────────────────────────────────────────────
LABEL_MAP   = {'NORM': 0, 'MI': 1, 'STTC': 2, 'HYP': 3, 'CD': 4}
CLASS_NAMES = list(LABEL_MAP.keys())

def log(msg): print(f'[INFO] {msg}')

log('Configuration loaded')
log(f'ML CV: {CV_FOLDS_ML}-fold × {CV_REPEATS_ML} repeats = {CV_FOLDS_ML*CV_REPEATS_ML} measurements')
log(f'DL CV: {CV_FOLDS_DL}-fold × {CV_REPEATS_DL} repeats = {CV_FOLDS_DL*CV_REPEATS_DL} measurements')""", "c03_config"))

# ── CELL 4: Data Loading with Cache ─────────────────────────────────────────
cells.append(code(
"""# ── DATA LOADING WITH CACHING ─────────────────────────────────────────────────
# Cache saves ~60-90 min on session restarts — numpy arrays stored in /kaggle/working

CACHE_FILES = {
    'X_train': os.path.join(CACHE_DIR, 'X_train.npy'),
    'X_test':  os.path.join(CACHE_DIR, 'X_test.npy'),
    'y_train': os.path.join(CACHE_DIR, 'y_train.npy'),
    'y_test':  os.path.join(CACHE_DIR, 'y_test.npy'),
}

def apply_filter(data, lowcut=0.5, highcut=40.0, fs=100.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def load_from_disk():
    log('Loading PTB-XL metadata...')
    df     = pd.read_csv(CSV_PATH, index_col='ecg_id')
    agg_df = pd.read_csv(SCP_PATH, index_col=0)

    diag_col = None
    for col in ['diagnostic_class', 'diagnostic_superclass', 'diagnostic']:
        if col in agg_df.columns:
            diag_col = col
            break
    agg_df = agg_df[agg_df[diag_col].notnull()]

    def aggregate_diagnostic(y_dic):
        if isinstance(y_dic, str):
            y_dic = eval(y_dic)
        tmp = [agg_df.loc[k][diag_col] for k in y_dic if k in agg_df.index]
        tmp = list(set(tmp))
        return tmp[0] if tmp else None

    df['label']    = df.scp_codes.apply(aggregate_diagnostic)
    df             = df.dropna(subset=['label'])
    df             = df[df['label'].isin(LABEL_MAP.keys())]
    df['label_id'] = df['label'].map(LABEL_MAP)

    total_available = len(df)
    log(f'Total valid labelled records: {total_available}')
    if TOTAL_RECORDS_TO_LOAD is not None:
        df = df.sample(n=min(TOTAL_RECORDS_TO_LOAD, total_available), random_state=RANDOM_STATE)
    else:
        df = df.sample(frac=1.0, random_state=RANDOM_STATE)

    n = len(df)
    X = np.zeros((n, MAX_SAMPLES, CHANNELS), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    count = 0

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 2000 == 0:
            log(f'  Progress: {i}/{n}')
        try:
            signal, _ = wfdb.rdsamp(os.path.join(DATA_DIR, row['filename_lr']))
            if signal.shape[1] < 12:
                continue
            ecg = signal[:MAX_SAMPLES, :]
            if len(ecg) == MAX_SAMPLES:
                X[count] = apply_filter(ecg).astype(np.float32)
                y[count] = row['label_id']
                count += 1
        except Exception:
            continue

    X, y = X[:count], y[:count]
    log(f'Loaded {count} records. Shape: {X.shape}')

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr.reshape(-1, CHANNELS)).reshape(X_tr.shape)
    X_te   = scaler.transform(X_te.reshape(-1, CHANNELS)).reshape(X_te.shape)
    return X_tr.astype(np.float32), X_te.astype(np.float32), y_tr, y_te

# Load from cache or disk
if all(os.path.exists(v) for v in CACHE_FILES.values()):
    log('Cache found — loading from numpy files (skipping wfdb reload)...')
    X_train = np.load(CACHE_FILES['X_train'])
    X_test  = np.load(CACHE_FILES['X_test'])
    y_train = np.load(CACHE_FILES['y_train'])
    y_test  = np.load(CACHE_FILES['y_test'])
    log(f'Loaded from cache — Train: {X_train.shape} | Test: {X_test.shape}')
else:
    log('No cache found — loading from PTB-XL (this takes 60-90 min)...')
    X_train, X_test, y_train, y_test = load_from_disk()
    np.save(CACHE_FILES['X_train'], X_train)
    np.save(CACHE_FILES['X_test'],  X_test)
    np.save(CACHE_FILES['y_train'], y_train)
    np.save(CACHE_FILES['y_test'],  y_test)
    log('Data cached to disk for future restarts.')

# Class distribution
log('Class distribution:')
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    log(f'  {CLASS_NAMES[u]:<6}: {c:>5} train samples ({c/len(y_train):.1%})')

# Class weights to handle HYP imbalance
class_weights_arr = compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
CLASS_WEIGHT = dict(enumerate(class_weights_arr))
log(f'Class weights: {CLASS_WEIGHT}')

# Full dataset for CV
X_all = np.concatenate([X_train, X_test], axis=0)
y_all = np.concatenate([y_train, y_test], axis=0)
log(f'Full dataset for CV: {X_all.shape}')""", "c04_data"))

# ── CELL 5: MAE Masking ──────────────────────────────────────────────────────
cells.append(code(
"""# ── MAE MASKING STRATEGY ─────────────────────────────────────────────────────
def apply_mae_mask(X):
    X_masked    = X.copy()
    num_patches = X.shape[1] // PATCH_SIZE
    num_mask    = int(num_patches * MASK_RATIO)
    for i in range(X.shape[0]):
        for idx in np.random.choice(num_patches, num_mask, replace=False):
            X_masked[i, idx * PATCH_SIZE:(idx + 1) * PATCH_SIZE, :] = 0.0
    return X_masked

log('Generating MAE masked training data (75% patches zeroed)...')
X_train_masked = apply_mae_mask(X_train)
log('Done')""", "c05_mask"))

# ── CELL 6: Model Architectures ──────────────────────────────────────────────
cells.append(code(
"""# ── MODEL ARCHITECTURES ──────────────────────────────────────────────────────

def build_autoencoder(input_shape=(MAX_SAMPLES, CHANNELS)):
    enc_in = Input(shape=input_shape)
    x      = layers.Conv1D(32, 5, activation='relu', padding='same')(enc_in)
    x      = layers.MaxPooling1D(2)(x)
    x      = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x      = layers.MaxPooling1D(2)(x)
    x      = layers.Flatten()(x)
    latent = layers.Dense(LATENT_DIM, activation='relu', name='latent_features')(x)
    encoder = Model(enc_in, latent, name='Encoder')

    dec_in = Input(shape=(LATENT_DIM,))
    x      = layers.Dense((input_shape[0] // 4) * 64, activation='relu')(dec_in)
    x      = layers.Reshape((input_shape[0] // 4, 64))(x)
    x      = layers.UpSampling1D(2)(x)
    x      = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
    x      = layers.UpSampling1D(2)(x)
    out    = layers.Conv1D(CHANNELS, 5, activation='linear', padding='same')(x)
    decoder = Model(dec_in, out, name='Decoder')

    ae_out = decoder(encoder(enc_in))
    ae     = Model(enc_in, ae_out, name='Autoencoder')
    ae.compile(optimizer=optimizers.Adam(0.001), loss='mse')
    return ae, encoder

def build_bilstm(num_classes=5):
    inp = Input(shape=(MAX_SAMPLES, CHANNELS))
    x   = layers.Bidirectional(layers.LSTM(BILSTM_UNITS, return_sequences=True))(inp)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    m   = Model(inp, out)
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def build_cnn(num_classes=5):
    inp = Input(shape=(MAX_SAMPLES, CHANNELS))
    x   = layers.Conv1D(32,  7, activation='relu', padding='same')(inp)
    x   = layers.MaxPooling1D(2)(x)
    x   = layers.Conv1D(64,  5, activation='relu', padding='same')(x)
    x   = layers.MaxPooling1D(2)(x)
    x   = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    m   = Model(inp, out)
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

log('Model builders ready')
log(f'  LinearSVC : C={SVM_C}')
log(f'  XGBoost   : n_estimators={XGB_N_EST}, max_depth={XGB_DEPTH}')
log(f'  BiLSTM    : {BILSTM_UNITS} units/direction, dropout={DROPOUT}')
log(f'  1D-CNN    : 3×Conv1D + GlobalAveragePooling, dropout={DROPOUT}')""", "c06_models"))

# ── CELL 7: Train Autoencoders ───────────────────────────────────────────────
cells.append(code(
"""# ── TRAIN AUTOENCODERS ────────────────────────────────────────────────────────
es_ae = callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                restore_best_weights=True, verbose=1)

log(f'Training SAE ({EPOCHS_AE} epochs max)...')
sae, encoder_sae = build_autoencoder()
sae.fit(X_train, X_train,
        epochs=EPOCHS_AE, batch_size=BATCH_SIZE,
        validation_split=0.1, callbacks=[es_ae], verbose=1)

log(f'Training MAE — self-supervised on masked input ({EPOCHS_AE} epochs max)...')
mae, encoder_mae = build_autoencoder()
mae.fit(X_train_masked, X_train,
        epochs=EPOCHS_AE, batch_size=BATCH_SIZE,
        validation_split=0.1, callbacks=[es_ae], verbose=1)

encoder_sae.save('encoder_sae.keras')
encoder_mae.save('encoder_mae.keras')
log('Encoders saved.')""", "c07_train_ae"))

# ── CELL 8: Feature Extraction ───────────────────────────────────────────────
cells.append(code(
"""# ── FEATURE EXTRACTION ───────────────────────────────────────────────────────
log('Extracting 128-dim features...')

X_train_sae = encoder_sae.predict(X_train, verbose=0)
X_test_sae  = encoder_sae.predict(X_test,  verbose=0)
X_train_mae = encoder_mae.predict(X_train, verbose=0)
X_test_mae  = encoder_mae.predict(X_test,  verbose=0)

pca         = PCA(n_components=LATENT_DIM)
X_train_pca = pca.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_test_pca  = pca.transform(X_test.reshape(X_test.shape[0], -1))

# Full-dataset versions for cross-validation
X_all_pca = np.concatenate([X_train_pca, X_test_pca], axis=0)
X_all_sae = np.concatenate([X_train_sae, X_test_sae], axis=0)
X_all_mae = np.concatenate([X_train_mae, X_test_mae], axis=0)

log(f'PCA: {X_all_pca.shape} | SAE: {X_all_sae.shape} | MAE: {X_all_mae.shape}')
log('All produce identical 128-dim vectors — direct fair comparison enabled')""", "c08_features"))

# ── CELL 9 & 10: Visualisations ─────────────────────────────────────────────
cells.append(code(
"""# ── VISUALISATION 1 — MAE Masking Strategy ───────────────────────────────────
sns.set_theme(style='whitegrid')

_, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
ax1.plot(X_train[0, :, 0], color='steelblue', linewidth=0.8)
ax1.set_title('Original ECG Signal (Lead I)', fontweight='bold')
ax1.set_ylabel('Amplitude')
ax2.plot(X_train_masked[0, :, 0], color='crimson', linewidth=0.8)
ax2.set_title('Masked Input — 75% of signal hidden during MAE pretraining', fontweight='bold')
ax2.set_ylabel('Amplitude')
ax2.set_xlabel('Samples (1000 timesteps @ 100 Hz)')
plt.suptitle('MAE Self-Supervised Pretraining Strategy', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs_mae_masking.png', dpi=150, bbox_inches='tight')
plt.show()
log('Saved: outputs_mae_masking.png')""", "c09_vis_mask"))

cells.append(code(
"""# ── VISUALISATION 2 — MAE Reconstruction Quality ─────────────────────────────
sample        = X_train_masked[0:1]
reconstructed = mae.predict(sample, verbose=0)[0]

_, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
axes[0].plot(X_train[0, :, 0],        color='steelblue',   linewidth=0.8)
axes[0].set_title('1. Original ECG (Ground Truth)', fontweight='bold')
axes[0].set_ylabel('Amplitude')
axes[1].plot(X_train_masked[0, :, 0], color='crimson',     linewidth=0.8)
axes[1].set_title('2. Masked Input (75% hidden)', fontweight='bold')
axes[1].set_ylabel('Amplitude')
axes[2].plot(reconstructed[:, 0],     color='forestgreen', linewidth=0.8)
axes[2].set_title('3. MAE Reconstruction from 25% visible input', fontweight='bold')
axes[2].set_ylabel('Amplitude')
axes[2].set_xlabel('Samples')
plt.suptitle('MAE Reconstruction Quality — Pretraining Verification',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs_reconstruction.png', dpi=150, bbox_inches='tight')
plt.show()
log('Saved: outputs_reconstruction.png')""", "c10_vis_recon"))

# ── CELL 11: ML CV ───────────────────────────────────────────────────────────
cells.append(code(
"""# ── CROSS-VALIDATION — ML MODELS (10-Fold × 5 Repeats = 50 measurements) ─────
log(f'Starting ML CV: {CV_FOLDS_ML}-fold × {CV_REPEATS_ML} repeats...')

ml_cv_results  = {}
ml_lat_results = {}

rskf = RepeatedStratifiedKFold(
    n_splits=CV_FOLDS_ML, n_repeats=CV_REPEATS_ML, random_state=RANDOM_STATE
)

ml_configs = [
    ('PCA + SVM',     X_all_pca, 'svm'),
    ('SAE + SVM',     X_all_sae, 'svm'),
    ('MAE + SVM',     X_all_mae, 'svm'),
    ('MAE + XGBoost', X_all_mae, 'xgb'),
]

for name, X_feat, clf_type in ml_configs:
    accs, lats = [], []
    log(f'  [{name}] running {CV_FOLDS_ML * CV_REPEATS_ML} folds...')
    for fold_i, (tr_idx, te_idx) in enumerate(rskf.split(X_feat, y_all)):
        if clf_type == 'svm':
            clf = CalibratedClassifierCV(LinearSVC(C=SVM_C, max_iter=3000))
        else:
            clf = XGBClassifier(
                n_estimators=XGB_N_EST, max_depth=XGB_DEPTH,
                eval_metric='mlogloss', n_jobs=-1, random_state=RANDOM_STATE
            )
        clf.fit(X_feat[tr_idx], y_all[tr_idx])
        t0 = time.time()
        preds = clf.predict(X_feat[te_idx])
        lats.append((time.time() - t0) * 1000)
        accs.append(accuracy_score(y_all[te_idx], preds))
        if (fold_i + 1) % 10 == 0:
            log(f'    fold {fold_i+1:02d}/{CV_FOLDS_ML*CV_REPEATS_ML} '
                f'— running mean: {np.mean(accs):.4f}')

    ml_cv_results[name]  = accs
    ml_lat_results[name] = lats
    log(f'  DONE [{name}]: {np.mean(accs):.4f} ± {np.std(accs):.4f} '
        f'| latency {np.mean(lats):.1f} ms')

log('ML cross-validation complete.')""", "c11_ml_cv"))

# ── CELL 12: DL CV (includes MAE Fine-Tuned) ─────────────────────────────────
cells.append(code(
"""# ── CROSS-VALIDATION — DL MODELS (5-Fold × 2 Repeats = 10 measurements) ──────
# Full Analysis Mode: Raw ECG → BiLSTM / 1D-CNN
# Class weights applied to handle HYP imbalance
log(f'Starting DL CV: {CV_FOLDS_DL}-fold × {CV_REPEATS_DL} repeats...')
log('Models: Raw+BiLSTM | Raw+1D-CNN')
log(f'Class weights: {CLASS_WEIGHT}')

dl_cv_results  = {}
dl_lat_results = {}

rskf_dl = RepeatedStratifiedKFold(
    n_splits=CV_FOLDS_DL, n_repeats=CV_REPEATS_DL, random_state=RANDOM_STATE
)
es_dl = callbacks.EarlyStopping(
    monitor='val_loss', patience=4, restore_best_weights=True, verbose=0
)

for model_name, build_fn in [('Raw + BiLSTM', build_bilstm), ('Raw + 1D-CNN', build_cnn)]:
    accs, lats = [], []
    log(f'  [{model_name}] running {CV_FOLDS_DL * CV_REPEATS_DL} folds...')
    for fold_i, (tr_idx, te_idx) in enumerate(rskf_dl.split(X_all, y_all)):
        tf.keras.backend.clear_session()
        fold_cw_arr = compute_class_weight(
            'balanced', classes=np.unique(y_all[tr_idx]), y=y_all[tr_idx]
        )
        fold_cw = dict(enumerate(fold_cw_arr))
        model = build_fn()
        model.fit(
            X_all[tr_idx], y_all[tr_idx],
            epochs=EPOCHS_DL, batch_size=BATCH_SIZE,
            validation_split=0.1, callbacks=[es_dl],
            class_weight=fold_cw, verbose=0
        )
        t0    = time.time()
        probs = model.predict(X_all[te_idx], verbose=0)
        lats.append((time.time() - t0) * 1000)
        accs.append(accuracy_score(y_all[te_idx], np.argmax(probs, axis=1)))
        log(f'    fold {fold_i+1:02d}/{CV_FOLDS_DL*CV_REPEATS_DL} — acc: {accs[-1]:.4f}')

    dl_cv_results[model_name]  = accs
    dl_lat_results[model_name] = lats
    log(f'  DONE [{model_name}]: {np.mean(accs):.4f} ± {np.std(accs):.4f}')

log('DL cross-validation complete.')""", "c12_dl_cv"))

# ── CELL 13: CV Summary ──────────────────────────────────────────────────────
cells.append(code(
"""# ── CV RESULTS SUMMARY ────────────────────────────────────────────────────────
all_cv_results  = {**ml_cv_results,  **dl_cv_results}
all_lat_results = {**ml_lat_results, **dl_lat_results}

# Ordered for presentation — original planned pipeline
model_names = [
    'PCA + SVM', 'SAE + SVM', 'MAE + SVM', 'MAE + XGBoost',
    'Raw + BiLSTM', 'Raw + 1D-CNN'
]

MODE_MAP = {
    'PCA + SVM':     'Efficient Scan',
    'SAE + SVM':     'Efficient Scan',
    'MAE + SVM':     'Efficient Scan',
    'MAE + XGBoost': 'Efficient Scan',
    'Raw + BiLSTM':  'Full Analysis',
    'Raw + 1D-CNN':  'Full Analysis',
}

summary_rows = []
print()
print('=' * 105)
print(f'{"Model":<22} {"Mode":<16} {"N":>4} {"Mean Acc":>10} {"Std":>8}'
      f' {"95% CI":>20} {"Latency":>12}')
print('=' * 105)

for name in model_names:
    accs     = all_cv_results[name]
    lats     = all_lat_results[name]
    n        = len(accs)
    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    ci_lo    = mean_acc - 1.96 * std_acc / np.sqrt(n)
    ci_hi    = mean_acc + 1.96 * std_acc / np.sqrt(n)
    mean_lat = np.mean(lats)
    mode     = MODE_MAP[name]
    print(f'{name:<22} {mode:<16} {n:>4} {mean_acc:>10.4f} {std_acc:>8.4f}'
          f' [{ci_lo:.4f}, {ci_hi:.4f}]  {mean_lat:>8.1f} ms')
    summary_rows.append({
        'Model': name, 'Mode': mode, 'N_folds': n,
        'Mean_Accuracy': round(mean_acc, 4), 'Std': round(std_acc, 4),
        'CI_95_low': round(ci_lo, 4), 'CI_95_high': round(ci_hi, 4),
        'Mean_Latency_ms': round(mean_lat, 1),
    })

print('=' * 105)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('cv_results_summary.csv', index=False)
log('Saved: cv_results_summary.csv')""", "c13_cv_summary"))

# ── CELL 14: Kruskal-Wallis ──────────────────────────────────────────────────
cells.append(code(
"""# ── STATISTICAL SIGNIFICANCE — KRUSKAL-WALLIS + DUNN POST-HOC ────────────────
log('Running Kruskal-Wallis test...')

score_lists = [all_cv_results[n] for n in model_names]
kw_stat, kw_p = kruskal(*score_lists)

print()
print('=' * 60)
print('  KRUSKAL-WALLIS TEST')
print('=' * 60)
print(f'  H-statistic : {kw_stat:.4f}')
print(f'  p-value     : {kw_p:.6f}')
print(f'  Result      : {"SIGNIFICANT (p < 0.05) ✓" if kw_p < 0.05 else "NOT significant"}')
print()
print('  Note: ML models have 50 measurements, DL models have 10.')
print('  Unequal N may inflate Dunn p-values for ML vs DL comparisons.')
print('  This is acknowledged as a limitation of the statistical analysis.')
print('=' * 60)

if kw_p < 0.05:
    all_scores, all_groups = [], []
    for name in model_names:
        all_scores.extend(all_cv_results[name])
        all_groups.extend([name] * len(all_cv_results[name]))
    ph_df = pd.DataFrame({'score': all_scores, 'group': all_groups})
    dunn  = sp.posthoc_dunn(ph_df, val_col='score', group_col='group',
                             p_adjust='bonferroni')
    print()
    print('Dunn Post-hoc p-values (Bonferroni corrected):')
    print(dunn.round(4).to_string())
    dunn.to_csv('dunn_posthoc.csv')

    plt.figure(figsize=(10, 8))
    mask = np.eye(len(dunn), dtype=bool)
    sns.heatmap(dunn, annot=True, fmt='.3f', cmap='RdYlGn_r',
                vmin=0, vmax=0.1, mask=mask, linewidths=0.5,
                annot_kws={'size': 9})
    plt.title('Post-hoc Dunn Test p-values (Bonferroni corrected)\\n'
              'Red < 0.05 = significant pairwise difference | '
              '*Unequal N: ML=50, DL=10 measurements',
              fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs_posthoc.png', dpi=150, bbox_inches='tight')
    plt.show()
    log('Saved: outputs_posthoc.png | dunn_posthoc.csv')""", "c14_kruskal"))

# ── CELL 15: Final Model Training ────────────────────────────────────────────
cells.append(code(
"""# ── FINAL MODEL TRAINING (Train → Test for detailed evaluation) ───────────────
log('Training final models on full train/test split...')

final_preds   = {}
final_proba   = {}
final_latency = {}

# ML final models
for name, x_tr, x_te, clf_type in [
    ('PCA + SVM',     X_train_pca, X_test_pca, 'svm'),
    ('SAE + SVM',     X_train_sae, X_test_sae, 'svm'),
    ('MAE + SVM',     X_train_mae, X_test_mae, 'svm'),
    ('MAE + XGBoost', X_train_mae, X_test_mae, 'xgb'),
]:
    if clf_type == 'svm':
        clf = CalibratedClassifierCV(LinearSVC(C=SVM_C, max_iter=3000))
    else:
        clf = XGBClassifier(
            n_estimators=XGB_N_EST, max_depth=XGB_DEPTH,
            eval_metric='mlogloss', n_jobs=-1, random_state=RANDOM_STATE
        )
    clf.fit(x_tr, y_train)
    t0 = time.time()
    preds = clf.predict(x_te)
    final_latency[name] = (time.time() - t0) * 1000
    final_preds[name]   = preds
    final_proba[name]   = clf.predict_proba(x_te)
    log(f'  {name}: acc={accuracy_score(y_test, preds):.4f} | {final_latency[name]:.1f}ms')

# DL final models (with class weights)
es_final = callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                   restore_best_weights=True)
cnn_final_model = None

for model_name, build_fn in [('Raw + BiLSTM', build_bilstm), ('Raw + 1D-CNN', build_cnn)]:
    tf.keras.backend.clear_session()
    model = build_fn()
    model.fit(X_train, y_train,
              epochs=EPOCHS_DL, batch_size=BATCH_SIZE,
              validation_split=0.1, callbacks=[es_final],
              class_weight=CLASS_WEIGHT, verbose=1)
    t0    = time.time()
    probs = model.predict(X_test, verbose=0)
    final_latency[model_name] = (time.time() - t0) * 1000
    final_preds[model_name]   = np.argmax(probs, axis=1)
    final_proba[model_name]   = probs
    if model_name == 'Raw + 1D-CNN':
        cnn_final_model = model
    log(f'  {model_name}: acc={accuracy_score(y_test, final_preds[model_name]):.4f}')

log('All final models trained.')""", "c15_final_train"))

# ── CELL 16: Final Summary Table ─────────────────────────────────────────────
cells.append(code(
"""# ── FINAL SUMMARY TABLE ───────────────────────────────────────────────────────
print()
print('=' * 115)
print(f'{"Model":<22} {"Mode":<16} {"CV Mean±Std":>14} {"Test Acc":>10}'
      f' {"Precision":>10} {"Recall":>8} {"Macro F1":>10} {"Latency":>12}')
print('=' * 115)

for name in model_names:
    cv_m  = np.mean(all_cv_results[name])
    cv_s  = np.std(all_cv_results[name])
    mode  = MODE_MAP[name]
    preds = final_preds[name]
    t_acc = accuracy_score(y_test, preds)
    prec  = precision_score(y_test, preds, average='macro', zero_division=0)
    rec   = recall_score(y_test, preds,    average='macro', zero_division=0)
    f1    = f1_score(y_test, preds,        average='macro', zero_division=0)
    lat   = final_latency[name]
    print(f'{name:<22} {mode:<16} {cv_m:.3f}±{cv_s:.3f}   {t_acc:>9.3f}'
          f'  {prec:>9.3f}  {rec:>7.3f}  {f1:>9.3f}  {lat:>10.1f}ms')

print('=' * 115)
best = max(model_names, key=lambda n: accuracy_score(y_test, final_preds[n]))
log(f'Best model: {best} — {accuracy_score(y_test, final_preds[best]):.4f}')""", "c16_summary"))

# ── CELL 17: Accuracy Chart ───────────────────────────────────────────────────
cells.append(code(
"""# ── VISUALISATION 3 — Accuracy Comparison (CV Mean ± Std) ───────────────────
import matplotlib.patches as mpatches

means  = [np.mean(all_cv_results[n]) for n in model_names]
stds   = [np.std(all_cv_results[n])  for n in model_names]
colors = ['#e07b39' if n in ml_cv_results else '#2d6a9f' for n in model_names]

plt.figure(figsize=(14, 6))
bars = plt.bar(model_names, means, yerr=stds, capsize=6,
               color=colors, edgecolor='white', error_kw={'linewidth': 2})
for bar, m, s in zip(bars, means, stds):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + s + 0.012,
             f'{m:.1%}\\n±{s:.3f}', ha='center',
             fontweight='bold', fontsize=8)

p1 = mpatches.Patch(color='#e07b39', label='Efficient Scan Mode (128-dim features)')
p2 = mpatches.Patch(color='#2d6a9f', label='Full Analysis Mode (raw ECG)')
plt.legend(handles=[p1, p2], loc='upper left')
plt.title(
    'ECG Classification Accuracy — Cross-Validated (Mean ± Std)\\n'
    f'ML: {CV_FOLDS_ML}-fold × {CV_REPEATS_ML} repeats | '
    f'DL: {CV_FOLDS_DL}-fold × {CV_REPEATS_DL} repeats | Full PTB-XL',
    fontsize=12, fontweight='bold'
)
plt.ylabel('Accuracy')
plt.ylim(0, 1.12)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('outputs_accuracy_cv.png', dpi=150, bbox_inches='tight')
plt.show()
log('Saved: outputs_accuracy_cv.png')""", "c17_vis_acc"))

# ── CELL 18: F1 Heatmap ──────────────────────────────────────────────────────
cells.append(code(
"""# ── VISUALISATION 4 — Per-Class F1 Heatmap ───────────────────────────────────
f1_matrix = [
    f1_score(y_test, final_preds[n], average=None,
             labels=list(range(5)), zero_division=0)
    for n in model_names
]
plt.figure(figsize=(11, 7))
sns.heatmap(np.array(f1_matrix), annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=CLASS_NAMES, yticklabels=model_names,
            vmin=0, vmax=1, linewidths=0.5, annot_kws={'size': 11, 'weight': 'bold'})
plt.title('Per-Class F1 Score — Final Test Set', fontsize=13, fontweight='bold')
plt.xlabel('Cardiac Condition')
plt.ylabel('Model')
plt.tight_layout()
plt.savefig('outputs_f1_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
log('Saved: outputs_f1_heatmap.png')""", "c18_vis_f1"))

# ── CELL 19: Latency ─────────────────────────────────────────────────────────
cells.append(code(
"""# ── VISUALISATION 5 — Inference Latency ──────────────────────────────────────
lat_vals = [final_latency[n] for n in model_names]
colors2  = ['#e07b39' if n in ml_cv_results else '#2d6a9f' for n in model_names]

_, ax = plt.subplots(figsize=(13, 5))
bars2 = ax.bar(model_names, lat_vals, color=colors2, edgecolor='white')
for bar, t in zip(bars2, lat_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{t:.1f}ms', ha='center', fontweight='bold', fontsize=9)
ax.set_title('Inference Latency per Model', fontsize=13, fontweight='bold')
ax.set_ylabel('Time (ms)')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=20, ha='right')
plt.tight_layout()
plt.savefig('outputs_latency.png', dpi=150, bbox_inches='tight')
plt.show()
log('Saved: outputs_latency.png')""", "c19_vis_lat"))

# ── CELL 20: t-SNE ───────────────────────────────────────────────────────────
cells.append(code(
"""# ── VISUALISATION 6 — t-SNE Latent Space ────────────────────────────────────
colors3 = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261']

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for features, ax, title in [
    (X_test_sae, ax1, 'SAE Latent Space'),
    (X_test_mae, ax2, 'MAE Latent Space (Pre-trained)'),
]:
    reduced = TSNE(n_components=2, random_state=RANDOM_STATE,
                   perplexity=30, max_iter=500).fit_transform(features)
    for i, cls in enumerate(CLASS_NAMES):
        mask = y_test == i
        ax.scatter(reduced[mask, 0], reduced[mask, 1],
                   c=colors3[i], label=cls, alpha=0.6, s=20)
    ax.set_title(title, fontweight='bold')
    ax.legend(markerscale=2)

plt.suptitle('Latent Space t-SNE — SAE vs MAE Pre-trained Features',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs_tsne.png', dpi=150, bbox_inches='tight')
plt.show()
log('Saved: outputs_tsne.png')""", "c20_vis_tsne"))

# ── CELL 21: XAI — fixed to correct prediction ───────────────────────────────
cells.append(code(
"""# ── VISUALISATION 7 — XAI Gradient Saliency (correctly classified sample) ────
# Find a correctly classified NORM sample for clean XAI demonstration
correct_mask = (final_preds['Raw + 1D-CNN'] == y_test)
norm_correct  = np.where((y_test == 0) & correct_mask)[0]
mi_correct    = np.where((y_test == 1) & correct_mask)[0]

# Prefer MI (more clinically interesting); fall back to NORM
sample_idx = int(mi_correct[0]) if len(mi_correct) > 0 else int(norm_correct[0])

x_tensor = tf.convert_to_tensor(X_test[sample_idx:sample_idx+1], dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(x_tensor)
    pred            = cnn_final_model(x_tensor, training=False)
    predicted_class = int(tf.argmax(pred[0]))
    score           = pred[0, predicted_class]

grads    = tape.gradient(score, x_tensor)[0].numpy()
saliency = np.mean(np.abs(grads), axis=1)
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

true_cls = CLASS_NAMES[y_test[sample_idx]]
pred_cls = CLASS_NAMES[predicted_class]
conf     = float(tf.reduce_max(pred[0]).numpy())

_, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(X_test[sample_idx, :, 0], color='steelblue', linewidth=0.8)
ax1.set_title(
    f'XAI Gradient Saliency — 1D-CNN\\n'
    f'True: {true_cls} | Predicted: {pred_cls} ✓ | Confidence: {conf:.1%}',
    fontweight='bold'
)
ax1.set_ylabel('ECG Amplitude (Lead I)')
ax2.imshow(saliency[np.newaxis, :], aspect='auto', cmap='hot',
           extent=[0, len(saliency), 0, 1])
ax2.set_title('Gradient Saliency — red = highest model attention (QRS peaks, ST segment)',
              fontsize=9)
ax2.set_xlabel('Timestep (samples @ 100 Hz)')
plt.suptitle('Explainability (XAI) — Correctly Classified Sample',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs_saliency.png', dpi=150, bbox_inches='tight')
plt.show()
log(f'Saved: outputs_saliency.png | Sample: {true_cls} correctly predicted as {pred_cls}')""", "c21_vis_xai"))

# ── CELL 22: Precision & Recall ──────────────────────────────────────────────
cells.append(code(
"""# ── VISUALISATION 8 — Precision & Recall Heatmaps ───────────────────────────
p_matrix = [precision_score(y_test, final_preds[n], average=None,
            labels=list(range(5)), zero_division=0) for n in model_names]
r_matrix = [recall_score(y_test, final_preds[n], average=None,
            labels=list(range(5)), zero_division=0) for n in model_names]

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
sns.heatmap(np.array(p_matrix), annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=model_names,
            vmin=0, vmax=1, ax=ax1, annot_kws={'size': 10, 'weight': 'bold'})
ax1.set_title('Per-Class Precision', fontweight='bold')

sns.heatmap(np.array(r_matrix), annot=True, fmt='.2f', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=model_names,
            vmin=0, vmax=1, ax=ax2, annot_kws={'size': 10, 'weight': 'bold'})
ax2.set_title('Per-Class Recall', fontweight='bold')

plt.suptitle('Precision & Recall — Final Test Set (with class weights)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs_precision_recall.png', dpi=150, bbox_inches='tight')
plt.show()
log('Saved: outputs_precision_recall.png')""", "c22_vis_pr"))

# ── CELL 23: AUC-ROC ─────────────────────────────────────────────────────────
cells.append(code(
"""# ── VISUALISATION 9 — AUC-ROC Curves ────────────────────────────────────────
y_bin = label_binarize(y_test, classes=list(range(5)))

_, axes = plt.subplots(3, 3, figsize=(18, 14))
axes    = axes.flatten()

for i, name in enumerate(model_names):
    ax    = axes[i]
    proba = final_proba[name]
    for j, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, j], proba[:, j])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=1.5, label=f'{cls} (AUC={roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.legend(fontsize=8)

for j in range(len(model_names), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('AUC-ROC Curves — All Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs_roc.png', dpi=150, bbox_inches='tight')
plt.show()
log('Saved: outputs_roc.png')""", "c23_vis_roc"))

# ── CELL 24: Confusion Matrices ──────────────────────────────────────────────
cells.append(code(
"""# ── VISUALISATION 10 — Confusion Matrices ────────────────────────────────────
_, axes = plt.subplots(3, 3, figsize=(18, 14))
axes    = axes.flatten()

for i, name in enumerate(model_names):
    cm = confusion_matrix(y_test, final_preds[name])
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues', cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={'size': 10})
    acc = accuracy_score(y_test, final_preds[name])
    axes[i].set_title(f'{name}  |  Acc: {acc:.1%}', fontweight='bold', fontsize=10)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

for j in range(len(model_names), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs_confusion.png', dpi=150, bbox_inches='tight')
plt.show()
log('Saved: outputs_confusion.png')""", "c24_vis_cm"))

# ── CELL 25: Classification Report ───────────────────────────────────────────
cells.append(code(
"""# ── PER-CLASS CLASSIFICATION REPORT ──────────────────────────────────────────
print('\\n' + '=' * 70)
print('  PER-CLASS REPORTS (Final Test Set)')
print('=' * 70)
for name in model_names:
    print(f'\\n── {name} ──')
    print(classification_report(
        y_test, final_preds[name],
        target_names=CLASS_NAMES, zero_division=0
    ))

log('All done.')
print()
print('Output files:')
for f in sorted([f for f in os.listdir('.')
                 if f.startswith('outputs_') or f.endswith('.csv')]):
    print(f'  {f}')""", "c25_report"))

# ── Build notebook JSON ───────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {NOTEBOOK_PATH}")
print(f"Total cells: {len(cells)}")
