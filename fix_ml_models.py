"""
Retrain the 4 ML classifiers (PCA+SVM, SAE+SVM, MAE+SVM, MAE+XGBoost)
using the locally saved encoders and training data.

This fixes the feature mismatch where the old pkl files were trained on
features from a different encoder run than the saved X_test feature arrays.

Run from the project root:
    python fix_ml_models.py
"""

import os
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

RESULTS_DIR = 'results'
MODELS_DIR  = 'models'
LATENT_DIM  = 128
SVM_C       = 1.0
RANDOM_STATE = 42


# ── Custom layer needed to load .keras encoder files ─────────────────────────

class _PosEmbed(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim   = embed_dim
        self.emb         = layers.Embedding(num_patches, embed_dim)

    def call(self, x):
        return x + self.emb(tf.range(self.num_patches))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'num_patches': self.num_patches, 'embed_dim': self.embed_dim})
        return cfg


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}')


# ── Step 1: Load data ─────────────────────────────────────────────────────────

log('Loading training and test arrays...')
X_train = np.load(os.path.join(RESULTS_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(RESULTS_DIR, 'y_train.npy'))
X_test  = np.load(os.path.join(RESULTS_DIR, 'X_test.npy'))
y_test  = np.load(os.path.join(RESULTS_DIR, 'y_test.npy'))
log(f'X_train: {X_train.shape} | X_test: {X_test.shape}')


# ── Step 2: Load encoders ─────────────────────────────────────────────────────

log('Loading SAE encoder...')
encoder_sae = tf.keras.models.load_model(
    os.path.join(RESULTS_DIR, 'encoder_sae.keras'),
    custom_objects={'_PosEmbed': _PosEmbed}
)
log('Loading MAE encoder...')
encoder_mae = tf.keras.models.load_model(
    os.path.join(RESULTS_DIR, 'encoder_mae.keras'),
    custom_objects={'_PosEmbed': _PosEmbed}
)


# ── Step 3: Extract features ──────────────────────────────────────────────────

log('Extracting SAE features from X_train (this takes a few minutes)...')
X_train_sae = encoder_sae.predict(X_train, verbose=1)

log('Extracting SAE features from X_test...')
X_test_sae  = encoder_sae.predict(X_test, verbose=1)

log('Extracting MAE features from X_train...')
X_train_mae = encoder_mae.predict(X_train, verbose=1)

log('Extracting MAE features from X_test...')
X_test_mae  = encoder_mae.predict(X_test, verbose=1)

log('Fitting PCA on X_train...')
pca         = PCA(n_components=LATENT_DIM, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train.reshape(len(X_train), -1))
X_test_pca  = pca.transform(X_test.reshape(len(X_test), -1))

log(f'Feature shape: {X_train_sae.shape}  (SAE / MAE / PCA all same)')

# Save updated test feature arrays (now aligned with these encoders)
log('Saving updated X_test feature arrays to results/...')
np.save(os.path.join(RESULTS_DIR, 'X_test_sae.npy'), X_test_sae)
np.save(os.path.join(RESULTS_DIR, 'X_test_mae.npy'), X_test_mae)
np.save(os.path.join(RESULTS_DIR, 'X_test_pca.npy'), X_test_pca)


# ── Step 4: Compute class weights ─────────────────────────────────────────────

class_weight_arr = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
CLASS_WEIGHT     = dict(enumerate(class_weight_arr))
log(f'Class weights: {CLASS_WEIGHT}')


# ── Step 5: Retrain and save ML classifiers ───────────────────────────────────

results = {}

for name, X_tr, X_te in [
    ('PCA + SVM', X_train_pca, X_test_pca),
    ('SAE + SVM', X_train_sae, X_test_sae),
    ('MAE + SVM', X_train_mae, X_test_mae),
]:
    log(f'Training {name}...')
    clf = CalibratedClassifierCV(
        LinearSVC(C=SVM_C, max_iter=3000, class_weight='balanced')
    )
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    acc   = accuracy_score(y_test, preds)
    results[name] = acc
    log(f'  {name}: test accuracy = {acc:.4f}  ({acc:.1%})')
    save_path = os.path.join(MODELS_DIR, f'model_{name.lower().replace(" + ", "_").replace(" ", "_")}.pkl')
    pickle.dump(clf, open(save_path, 'wb'))
    log(f'  Saved: {save_path}')

log('Training MAE + XGBoost...')
sw  = np.array([class_weight_arr[c] for c in y_train])
xgb = XGBClassifier(
    n_estimators=100, max_depth=6,
    eval_metric='mlogloss', n_jobs=-1, random_state=RANDOM_STATE
)
xgb.fit(X_train_mae, y_train, sample_weight=sw)
preds = xgb.predict(X_test_mae)
acc   = accuracy_score(y_test, preds)
results['MAE + XGBoost'] = acc
log(f'  MAE + XGBoost: test accuracy = {acc:.4f}  ({acc:.1%})')
pickle.dump(xgb, open(os.path.join(MODELS_DIR, 'model_mae_xgboost.pkl'), 'wb'))
log(f'  Saved: models/model_mae_xgboost.pkl')


# ── Step 6: Summary ───────────────────────────────────────────────────────────

print()
print('=' * 50)
print('  RETRAINED ML MODEL RESULTS (Test Set)')
print('=' * 50)
for name, acc in results.items():
    print(f'  {name:<20}  {acc:.4f}  ({acc:.1%})')
print('=' * 50)
print()
print('All pkl files updated in models/')
print('All X_test feature arrays updated in results/')
print('Done.')
