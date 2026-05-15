import numpy as np, pickle, matplotlib, matplotlib.pyplot as plt
import seaborn as sns, tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

CLASS_NAMES  = ['NORM', 'MI', 'STTC', 'HYP', 'CD']
MODEL_NAMES  = ['PCA + SVM','SAE + SVM','MAE + SVM','MAE + XGBoost',
                'SAE Fine-tuned','MAE Fine-tuned','Raw + BiLSTM','Raw + 1D-CNN']
OUT          = 'results'   # figures saved here, matching existing project layout

class _PosEmbed(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kw):
        super().__init__(**kw)
        self.num_patches = num_patches; self.embed_dim = embed_dim
        self.emb = layers.Embedding(num_patches, embed_dim)
    def call(self, x): return x + self.emb(tf.range(self.num_patches))
    def get_config(self):
        c = super().get_config()
        c.update({'num_patches': self.num_patches, 'embed_dim': self.embed_dim})
        return c

co = {'_PosEmbed': _PosEmbed}

y_test     = np.load('results/y_test.npy')
X_test     = np.load('results/X_test.npy')
X_test_pca = np.load('results/X_test_pca.npy')
X_test_sae = np.load('results/X_test_sae.npy')
X_test_mae = np.load('results/X_test_mae.npy')
y_bin      = label_binarize(y_test, classes=list(range(5)))

MODELS_CFG = [
    ('PCA + SVM',      'pkl',   'models/model_pca_svm.pkl',         X_test_pca),
    ('SAE + SVM',      'pkl',   'models/model_sae_svm.pkl',         X_test_sae),
    ('MAE + SVM',      'pkl',   'models/model_mae_svm.pkl',         X_test_mae),
    ('MAE + XGBoost',  'pkl',   'models/model_mae_xgboost.pkl',     X_test_mae),
    ('SAE Fine-tuned', 'keras', 'models/model_sae_finetuned.keras', X_test),
    ('MAE Fine-tuned', 'keras', 'models/model_mae_finetuned.keras', X_test),
    ('Raw + BiLSTM',   'keras', 'models/model_bilstm.keras',        X_test),
    ('Raw + 1D-CNN',   'keras', 'models/model_cnn.keras',           X_test),
]

all_preds = {}; all_probs = {}
for name, ftype, path, X in MODELS_CFG:
    print(f'Loading {name}...')
    if ftype == 'pkl':
        m = pickle.load(open(path, 'rb'))
        all_preds[name] = m.predict(X)
        all_probs[name] = m.predict_proba(X)
    else:
        m = tf.keras.models.load_model(path, custom_objects=co)
        p = m.predict(X, verbose=0)
        all_preds[name] = np.argmax(p, axis=1)
        all_probs[name] = p
print('All models loaded.\n')

# ── Figure 1 — Confusion Matrices ─────────────────────────────────────────────
fig, axes = plt.subplots(4, 2, figsize=(14, 22))
fig.suptitle('Confusion Matrices — All Models', fontsize=16, fontweight='bold', y=1.01)
axes = axes.flatten()
for idx, name in enumerate(MODEL_NAMES):
    ax  = axes[idx]
    cm  = confusion_matrix(y_test, all_preds[name])
    acc = np.trace(cm) / cm.sum()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, cbar=False, annot_kws={'size': 8})
    ax.set_title(f'{name}  |  Acc: {acc:.1%}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('Actual', fontsize=9)
    ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/outputs_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print('DONE: outputs_confusion.png saved')

# ── Figure 2 — F1 Heatmap ─────────────────────────────────────────────────────
f1_matrix = np.zeros((8, 5))
for i, name in enumerate(MODEL_NAMES):
    f1_matrix[i] = f1_score(y_test, all_preds[name], average=None,
                             labels=list(range(5)), zero_division=0)
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(f1_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='F1 Score')
ax.set_xticks(range(5)); ax.set_yticks(range(8))
ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_yticklabels(MODEL_NAMES, fontsize=10)
ax.set_xlabel('Cardiac Condition', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
ax.set_title('Per-Class F1 Score — Final Test Set', fontsize=13, fontweight='bold')
for i in range(8):
    for j in range(5):
        v = f1_matrix[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='white' if v < 0.35 or v > 0.75 else 'black')
plt.tight_layout()
plt.savefig(f'{OUT}/outputs_f1_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('DONE: outputs_f1_heatmap.png saved')

# ── Figure 3 — Precision / Recall Heatmaps ────────────────────────────────────
prec_matrix = np.zeros((8, 5))
rec_matrix  = np.zeros((8, 5))
for i, name in enumerate(MODEL_NAMES):
    prec_matrix[i] = precision_score(y_test, all_preds[name], average=None,
                                     labels=list(range(5)), zero_division=0)
    rec_matrix[i]  = recall_score(y_test, all_preds[name], average=None,
                                  labels=list(range(5)), zero_division=0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Precision & Recall — Final Test Set (with class weights)',
             fontsize=13, fontweight='bold')
for ax, mat, title, cmap, label in [
    (ax1, prec_matrix, 'Per-Class Precision', 'Blues',  'Precision'),
    (ax2, rec_matrix,  'Per-Class Recall',    'Greens', 'Recall'),
]:
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label=label)
    ax.set_xticks(range(5)); ax.set_yticks(range(8))
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_yticklabels(MODEL_NAMES, fontsize=9)
    ax.set_xlabel('Cardiac Condition', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    for i in range(8):
        for j in range(5):
            v = mat[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if v > 0.75 else 'black')
plt.tight_layout()
plt.savefig(f'{OUT}/outputs_precision_recall.png', dpi=150, bbox_inches='tight')
plt.close()
print('DONE: outputs_precision_recall.png saved')

# ── Figure 4 — ROC Curves ─────────────────────────────────────────────────────
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
fig, axes = plt.subplots(4, 2, figsize=(14, 22))
fig.suptitle('AUC-ROC Curves — All Models', fontsize=16, fontweight='bold', y=1.01)
axes = axes.flatten()
for idx, name in enumerate(MODEL_NAMES):
    ax   = axes[idx]
    prob = all_probs[name]
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    for c in range(5):
        fpr, tpr, _ = roc_curve(y_bin[:, c], prob[:, c])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[c], lw=1.5,
                label=f'{CLASS_NAMES[c]} (AUC={roc_auc:.2f})')
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xlabel('FPR', fontsize=8)
    ax.set_ylabel('TPR', fontsize=8)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.tick_params(labelsize=7)
plt.tight_layout()
plt.savefig(f'{OUT}/outputs_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print('DONE: outputs_roc.png saved')

print('\nAll 4 figures regenerated successfully.')
