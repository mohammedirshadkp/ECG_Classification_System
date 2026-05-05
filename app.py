import os
import time
import pickle
from collections import Counter

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from scipy.signal import butter, filtfilt

# ── Class metadata ────────────────────────────────────────────────────────────

CLASS_NAMES = ['NORM', 'MI', 'STTC', 'HYP', 'CD']
LEAD_NAMES  = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

PLAIN_NAMES = {
    'NORM': 'Normal Heart Rhythm',
    'MI':   'Heart Attack Pattern',
    'STTC': 'ST/T Wave Abnormality',
    'HYP':  'Heart Enlargement',
    'CD':   'Conduction Problem',
}

PLAIN_DESCRIPTIONS = {
    'NORM': (
        'The ECG looks healthy. The heart appears to be beating normally '
        'with no detected signs of disease or structural abnormality.'
    ),
    'MI': (
        'Patterns associated with a heart attack (myocardial infarction) were detected. '
        'This means part of the heart muscle may not be receiving enough blood — '
        'which can indicate a current or past heart attack.'
    ),
    'STTC': (
        'Irregular patterns were found in the ST or T-wave parts of the ECG. '
        'This can be caused by heart stress, reduced blood supply, medication '
        'effects, or early coronary artery disease.'
    ),
    'HYP': (
        'The ECG shows signs that the heart muscle may be thicker than normal. '
        'This commonly happens when the heart has to work harder over a long '
        'period — often due to high blood pressure.'
    ),
    'CD': (
        "The heart's electrical system shows a conduction disturbance — "
        "meaning the electrical signals that coordinate heartbeats are not "
        "traveling through the heart in the usual pattern."
    ),
}

NEXT_STEPS = {
    'NORM': 'No immediate action required. Continue with regular annual health check-ups.',
    'MI':   'Seek urgent medical evaluation. A cardiologist should assess this finding as soon as possible.',
    'STTC': 'Schedule a consultation with a doctor. Further tests (stress test, blood work) may be needed.',
    'HYP':  'Consult a cardiologist to evaluate blood pressure and overall heart function.',
    'CD':   'See a cardiologist. Additional monitoring such as a Holter monitor may be recommended.',
}

SEVERITY       = {'NORM': 'normal', 'MI': 'critical', 'STTC': 'moderate', 'HYP': 'moderate', 'CD': 'moderate'}
SEVERITY_COLOR = {'normal': '#27ae60', 'moderate': '#e67e22', 'critical': '#c0392b'}
SEVERITY_EMOJI = {'normal': '✅', 'moderate': '⚠️', 'critical': '🚨'}
SEVERITY_BADGE = {'normal': 'No Concern', 'moderate': 'Needs Attention', 'critical': 'Serious Concern'}

# Validated cross-validation accuracy from PTB-XL test results
MODEL_ACCURACY = {
    'PCA + SVM':      0.4238,
    'SAE + SVM':      0.5379,
    'MAE + SVM':      0.5467,
    'MAE + XGBoost':  0.5310,
    'SAE Fine-tuned': 0.6416,
    'MAE Fine-tuned': 0.6560,
    'Raw + BiLSTM':   0.6804,
    'Raw + 1D-CNN':   0.7265,
}

def acc_label(acc):
    if acc >= 0.70: return ('Strong',  '#27ae60')
    if acc >= 0.60: return ('Good',    '#2980b9')
    if acc >= 0.50: return ('Moderate','#e67e22')
    return                  ('Limited', '#c0392b')

MODELS_DIR = 'models'
DATA_DIR   = 'results'

# ── Tier definitions — user-friendly labels ───────────────────────────────────
# The 4 ML classifiers (PCA/SAE/MAE+SVM, MAE+XGBoost) were evaluated in the
# research and their results are included in the validation metrics below.
# They are excluded from live prediction because their accuracy (42–55%) is
# too low to provide useful clinical guidance compared to the DL models (64–73%).

TIER_MODELS = {
    'End-to-End Deep Learning  (BiLSTM & 1D-CNN)':
        ['Raw + BiLSTM', 'Raw + 1D-CNN'],
    'Pre-trained + Fine-tuned  (SAE & MAE)':
        ['SAE Fine-tuned', 'MAE Fine-tuned'],
    'Compare All 4 Models':
        ['SAE Fine-tuned', 'MAE Fine-tuned', 'Raw + BiLSTM', 'Raw + 1D-CNN'],
}

TIER_INFO = {
    'End-to-End Deep Learning  (BiLSTM & 1D-CNN)': (
        'The complete ECG waveform is fed **directly into a neural network** with no '
        'summarisation step. The network learns on its own which patterns matter.  \n\n'
        'Think of it like a doctor who reads the entire ECG printout from start to finish '
        'rather than looking at a summary.  \n\n'
        'This is the highest-accuracy approach in the thesis. Two neural network types are '
        'compared — **BiLSTM** (sequential pattern reader) and **1D-CNN** (pattern detector). '
        'Takes about 2–4 seconds per ECG.'
    ),
    'Pre-trained + Fine-tuned  (SAE & MAE)': (
        'The model first learns general ECG patterns without using any diagnosis labels '
        '(unsupervised pre-training), then is further trained to classify the five heart '
        'conditions specifically (fine-tuning).  \n\n'
        'This is a transfer learning approach — the encoder builds general ECG knowledge '
        'first, then specialises for the 5 conditions. About 3 seconds per ECG.'
    ),
    'Compare All 4 Models': (
        'Runs all 4 models at once and shows every result side by side. '
        'The overall assessment is based on majority vote across models.  \n\n'
        'These are the top-performing models from the full 8-model research experiment — '
        'accuracy range **64% – 73%**. The complete 8-model comparison is shown in the '
        'validation metrics section below.'
    ),
}

MODEL_FULL_NAME = {
    'PCA + SVM':      'PCA + Support Vector Machine',
    'SAE + SVM':      'Sparse Autoencoder + SVM',
    'MAE + SVM':      'Masked Autoencoder + SVM',
    'MAE + XGBoost':  'Masked Autoencoder + XGBoost',
    'SAE Fine-tuned': 'Sparse Autoencoder (Fine-tuned)',
    'MAE Fine-tuned': 'Masked Autoencoder (Fine-tuned)',
    'Raw + BiLSTM':   'Bidirectional LSTM',
    'Raw + 1D-CNN':   '1D Convolutional Neural Network',
}

MODEL_TYPE = {
    'PCA + SVM':      'Traditional ML',
    'SAE + SVM':      'Traditional ML',
    'MAE + SVM':      'Traditional ML',
    'MAE + XGBoost':  'Traditional ML',
    'SAE Fine-tuned': 'Deep Learning',
    'MAE Fine-tuned': 'Deep Learning',
    'Raw + BiLSTM':   'Deep Learning',
    'Raw + 1D-CNN':   'Deep Learning',
}

MODEL_FILES = {
    'PCA + SVM':      ('pkl',   'model_pca_svm.pkl'),
    'SAE + SVM':      ('pkl',   'model_sae_svm.pkl'),
    'MAE + SVM':      ('pkl',   'model_mae_svm.pkl'),
    'MAE + XGBoost':  ('pkl',   'model_mae_xgboost.pkl'),
    'SAE Fine-tuned': ('keras', 'model_sae_finetuned.keras'),
    'MAE Fine-tuned': ('keras', 'model_mae_finetuned.keras'),
    'Raw + BiLSTM':   ('keras', 'model_bilstm.keras'),
    'Raw + 1D-CNN':   ('keras', 'model_cnn.keras'),
}

FEATURE_SOURCE = {
    'PCA + SVM':      'pca',
    'SAE + SVM':      'sae',
    'MAE + SVM':      'mae',
    'MAE + XGBoost':  'mae',
    'SAE Fine-tuned': 'raw',
    'MAE Fine-tuned': 'raw',
    'Raw + BiLSTM':   'raw',
    'Raw + 1D-CNN':   'raw',
}


# ── Custom Keras layer ────────────────────────────────────────────────────────

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


# ── Resource loading ──────────────────────────────────────────────────────────

@st.cache_resource
def load_test_data():
    required = {
        'X_test':     os.path.join(DATA_DIR, 'X_test.npy'),
        'y_test':     os.path.join(DATA_DIR, 'y_test.npy'),
        'X_test_pca': os.path.join(DATA_DIR, 'X_test_pca.npy'),
        'X_test_sae': os.path.join(DATA_DIR, 'X_test_sae.npy'),
        'X_test_mae': os.path.join(DATA_DIR, 'X_test_mae.npy'),
    }
    missing = [k for k, v in required.items() if not os.path.exists(v)]
    if missing:
        st.error(f'Missing files in results/: {missing}')
        st.stop()
    return {k: np.load(v) for k, v in required.items()}


@st.cache_resource
def load_models():
    loaded = {}
    for name, (ftype, fname) in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            continue
        try:
            if ftype == 'pkl':
                with open(path, 'rb') as f:
                    loaded[name] = pickle.load(f)
            else:
                loaded[name] = tf.keras.models.load_model(
                    path, custom_objects={'_PosEmbed': _PosEmbed}
                )
        except Exception:
            pass
    return loaded


@st.cache_resource
def load_encoders():
    encoders = {}
    for key, fname in [('sae', 'encoder_sae.keras'), ('mae', 'encoder_mae.keras')]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            try:
                encoders[key] = tf.keras.models.load_model(
                    path, custom_objects={'_PosEmbed': _PosEmbed}
                )
            except Exception:
                pass
    return encoders


# ── PTB-XL local dataset helpers ─────────────────────────────────────────────

@st.cache_data
def load_ptbxl_db():
    import pandas as pd
    import ast

    db = pd.read_csv(os.path.join('data', 'ptbxl_database.csv'), index_col='ecg_id')

    scp_stmt   = pd.read_csv(os.path.join('data', 'scp_statements.csv'), index_col=0)
    code_map   = (scp_stmt[scp_stmt['diagnostic'] == 1.0]['diagnostic_class']
                  .to_dict())
    return db, code_map


def resolve_label(scp_codes_str, code_map):
    import ast
    try:
        scp = ast.literal_eval(scp_codes_str)
    except Exception:
        return None
    scores = {}
    for code, conf in scp.items():
        cls = code_map.get(code)
        if cls in CLASS_NAMES:
            scores[cls] = scores.get(cls, 0) + conf
    return max(scores, key=scores.get) if scores else None


def load_wfdb_signal(filename_lr):
    import wfdb
    path   = os.path.join('data', filename_lr)
    record = wfdb.rdrecord(path)
    return record.p_signal.astype(np.float32)   # shape (1000, 12)

def translate_report(text):
    if not text or text.strip() in ('', 'nan', 'N/A'):
        return None
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source='de', target='en').translate(text)
    except Exception:
        return None


# ── ECG preprocessing ─────────────────────────────────────────────────────────

def bandpass_filter(ecg, fs=100, low=0.5, high=40.0):
    b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, ecg, axis=0)

def normalize_per_lead(ecg):
    mean = ecg.mean(axis=0, keepdims=True)
    std  = ecg.std(axis=0, keepdims=True) + 1e-8
    return (ecg - mean) / std

def preprocess_uploaded(raw_array):
    ecg = bandpass_filter(raw_array.astype(np.float32))
    return normalize_per_lead(ecg)


# ── Prediction ────────────────────────────────────────────────────────────────

def _infer(model, x):
    t0    = time.time()
    proba = (model.predict_proba(x)[0] if hasattr(model, 'predict_proba')
             else model.predict(x, verbose=0)[0])
    return proba, (time.time() - t0) * 1000

def predict_from_test(model_name, model, idx, data):
    src  = FEATURE_SOURCE[model_name]
    x    = data['X_test'][idx:idx+1] if src == 'raw' else data[f'X_test_{src}'][idx:idx+1]
    proba, lat = _infer(model, x)
    pred_idx   = int(np.argmax(proba))
    return CLASS_NAMES[pred_idx], float(proba[pred_idx]), lat, proba

def predict_from_upload(model_name, model, ecg, encoders):
    src = FEATURE_SOURCE[model_name]
    if src == 'raw':
        x = ecg[np.newaxis]
    elif src in encoders:
        feat = encoders[src].predict(ecg[np.newaxis], verbose=0)
        x    = feat.reshape(1, -1)
    else:
        return None, None, None, None
    proba, lat = _infer(model, x)
    pred_idx   = int(np.argmax(proba))
    return CLASS_NAMES[pred_idx], float(proba[pred_idx]), lat, proba


# ── UI helpers ────────────────────────────────────────────────────────────────

def conf_label(c):
    if c >= 0.85: return 'Very High'
    if c >= 0.70: return 'High'
    if c >= 0.55: return 'Moderate'
    return 'Low'

def health_card(pred, conf, agreement, n_models, true_label=None):
    sev   = SEVERITY[pred]
    color = SEVERITY_COLOR[sev]
    emoji = SEVERITY_EMOJI[sev]
    badge = SEVERITY_BADGE[sev]
    agree_pct = int(agreement / n_models * 100)

    st.markdown(f"""
    <div style="background:{color}15; border-left:6px solid {color};
                border-radius:10px; padding:22px 26px; margin:10px 0 18px 0">
      <div style="font-size:1.9rem; font-weight:700; color:{color}; margin-bottom:6px">
        {emoji}&nbsp; {PLAIN_NAMES[pred]}
      </div>
      <div style="font-size:1rem; color:#333; line-height:1.6">
        {PLAIN_DESCRIPTIONS[pred]}
      </div>
      <div style="margin-top:14px; padding:10px 14px; background:{color}22;
                  border-radius:6px; font-size:0.95rem; color:#333">
        <b>What to do:</b>&nbsp; {NEXT_STEPS[pred]}
      </div>
      <div style="margin-top:12px; font-size:0.85rem; color:#666; display:flex; gap:24px; flex-wrap:wrap">
        <span>Prediction confidence: <b>{conf:.0%}</b> ({conf_label(conf)})</span>
        <span>Model agreement: <b>{agreement} of {n_models} models ({agree_pct}%)</b></span>
        <span>Severity: <b style="color:{color}">{badge}</b></span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if true_label is not None:
        correct = pred == true_label
        icon    = '✅ Matches known diagnosis' if correct else '❌ Does not match known diagnosis'
        st.caption(
            f"Known diagnosis in this dataset sample: **{PLAIN_NAMES[true_label]}**"
            f"  —  {icon}"
        )

def plot_ecg(ecg, title='12-Lead ECG'):
    fig, axes = plt.subplots(3, 4, figsize=(16, 4.5), sharex=True)
    axes = axes.flatten()
    for i in range(12):
        axes[i].plot(ecg[:, i], linewidth=0.7, color='#2c7bb6')
        axes[i].set_title(LEAD_NAMES[i], fontsize=9, fontweight='bold')
        axes[i].set_yticks([])
        axes[i].grid(True, alpha=0.2)
    plt.suptitle(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig

def prob_bar(proba, pred):
    # Use short class codes as labels — avoids text overlap
    short_labels = ['Normal', 'Heart\nAttack', 'ST/T\nWave', 'Enlarge-\nment', 'Conduction\nProblem']
    colors = [SEVERITY_COLOR[SEVERITY[c]] if c == pred else '#d0d0d0' for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.bar(range(5), proba, color=colors, edgecolor='none', width=0.6)
    ax.set_xticks(range(5))
    ax.set_xticklabels(short_labels, fontsize=6.5, ha='center')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability', fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_title('Probability per condition', fontsize=8, pad=6)
    plt.tight_layout(pad=1.2)
    return fig


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title='ECG Heart Analyzer', layout='wide', page_icon='🫀')
st.title('🫀 ECG Heart Analyzer')
st.caption('Research tool  ·  PTB-XL Dataset  ·  VMU 2026  ·  Mohammed Irshad K P')

st.info(
    '⚕️ **Medical Disclaimer** — This is a research prototype for educational purposes only. '
    'Results must not replace professional medical advice, diagnosis, or treatment.'
)

models   = load_models()
encoders = load_encoders()

if not models:
    st.error('No models found in models/ folder. Add .pkl and .keras files and restart.')
    st.stop()

# ── Step 1 — ECG Input ────────────────────────────────────────────────────────

st.divider()
st.subheader('Step 1 — Provide an ECG Recording')

input_mode = st.radio(
    'How would you like to provide the ECG?',
    ['📂  Browse processed test set  (2,156 pre-labelled samples)',
     '🗂️  Pick from local PTB-XL dataset  (all 21,837 records)'],
    horizontal=True,
)

ecg_signal  = None
true_label  = None
patient_info = None
data        = None

# ── Mode A: processed test set ────────────────────────────────────────────────
if '📂' in input_mode:
    data = load_test_data()
    n    = len(data['X_test'])

    c1, c2 = st.columns([4, 1])
    with c2:
        if st.button('🎲 Pick Random'):
            st.session_state['idx'] = int(np.random.randint(0, n))
    with c1:
        idx = st.slider('Sample index', 0, n - 1,
                        st.session_state.get('idx', 0), key='slider')
        st.session_state['idx'] = idx

    ecg_signal = data['X_test'][idx]
    true_label = CLASS_NAMES[int(data['y_test'][idx])]
    st.success(
        f'Sample **#{idx}** loaded  —  '
        f'Recorded diagnosis: **{PLAIN_NAMES[true_label]}**'
    )

# ── Mode B: local PTB-XL raw records ─────────────────────────────────────────
elif '🗂️' in input_mode:
    db, code_map = load_ptbxl_db()

    st.markdown(
        'Enter any ECG ID from **1 to 21,837** to load that person\'s recording '
        'directly from the PTB-XL dataset stored in your `data/` folder.'
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        ecg_id = st.number_input('ECG ID', min_value=1, max_value=21837,
                                 value=st.session_state.get('ecg_id', 1),
                                 step=1, key='ecg_id_input')
        st.session_state['ecg_id'] = int(ecg_id)
    with c2:
        if st.button('🎲 Random person'):
            st.session_state['ecg_id'] = int(np.random.randint(1, 21838))
            st.rerun()

    ecg_id = int(st.session_state.get('ecg_id', 1))

    if ecg_id in db.index:
        row = db.loc[ecg_id]
        try:
            raw        = load_wfdb_signal(row['filename_lr'])
            ecg_signal = preprocess_uploaded(raw)
            true_label = resolve_label(row['scp_codes'], code_map)
            sex_str    = 'Male' if row['sex'] == 1 else 'Female'
            age_str    = f"{int(row['age'])} years old" if not np.isnan(row['age']) else 'Age unknown'
            patient_info = {
                'ecg_id':  ecg_id,
                'age':     age_str,
                'sex':     sex_str,
                'report':  str(row.get('report', '')),
                'label':   true_label,
            }

            label_txt = PLAIN_NAMES[true_label] if true_label else 'Unknown'
            st.success(
                f'ECG **#{ecg_id}** loaded  —  '
                f'{sex_str}, {age_str}  —  '
                f'Recorded diagnosis: **{label_txt}**'
            )

            with st.expander("📋 Patient record details"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f'**ECG ID:** {ecg_id}')
                    st.markdown(f'**Age:** {age_str}')
                    st.markdown(f'**Sex:** {sex_str}')
                with col_b:
                    st.markdown(f'**Mapped diagnosis:** {label_txt}')
                    report_raw = str(row.get('report', ''))
                    st.markdown(f"**Doctor's report (German):** _{report_raw}_")
                    translated = translate_report(report_raw)
                    if translated:
                        st.markdown(f"**Doctor's report (English):** _{translated}_")
                    else:
                        st.caption('Translation unavailable — check internet connection.')

        except Exception as e:
            st.error(f'Could not load ECG #{ecg_id}: {e}')
    else:
        st.error(f'ECG ID {ecg_id} not found in the database.')


if ecg_signal is None:
    st.info('Waiting for an ECG input to continue…')
    st.stop()

# ── Step 2 — ECG display ──────────────────────────────────────────────────────

st.divider()
st.subheader('Step 2 — ECG Signal Preview')
st.caption(
    'Each panel shows the electrical activity of the heart seen from a different '
    'angle (called a "lead"). Together these 12 views give a complete picture.'
)

if '📂' in input_mode:
    title_str = f'Sample #{idx}  —  True diagnosis: {PLAIN_NAMES[true_label]}'
else:
    label_txt = PLAIN_NAMES[patient_info['label']] if patient_info and patient_info['label'] else 'Unknown'
    title_str = (f"ECG #{patient_info['ecg_id']}  —  "
                 f"{patient_info['sex']}, {patient_info['age']}  —  {label_txt}"
                 if patient_info else 'ECG Recording')
fig = plot_ecg(ecg_signal, title_str)
st.pyplot(fig)
plt.close()

# ── Step 3 — Analysis type ────────────────────────────────────────────────────

st.divider()
st.subheader('Step 3 — Choose Deployment Mode')
st.caption(
    'The 4 best-performing models from the thesis experiment are available for live prediction. '
    'Full results for all 8 models (including the feature-extraction baselines) '
    'are shown in the validation metrics section below.'
)

tier = st.radio('', list(TIER_MODELS.keys()), label_visibility='collapsed')
st.info(TIER_INFO[tier])

# Sub-filter labels — what each model's pipeline actually is
MODEL_PIPELINE = {
    'PCA + SVM':      ('PCA  →  SVM',      'Linear compression → SVM classifier\n(baseline)'),
    'SAE + SVM':      ('SAE  →  SVM',      'Sparse autoencoder features → SVM classifier'),
    'MAE + SVM':      ('MAE  →  SVM',      'Masked autoencoder features → SVM classifier\n(thesis contribution)'),
    'MAE + XGBoost':  ('MAE  →  XGBoost',  'Masked autoencoder features → XGBoost classifier'),
    'Raw + BiLSTM':   ('Raw  →  BiLSTM',   'No extraction → Bidirectional LSTM (sequential)'),
    'Raw + 1D-CNN':   ('Raw  →  1D-CNN',   'No extraction → Convolutional Neural Network\n(highest accuracy)'),
    'SAE Fine-tuned': ('SAE fine-tuned',   'SAE encoder + classification head, end-to-end'),
    'MAE Fine-tuned': ('MAE fine-tuned',   'MAE encoder + classification head, end-to-end'),
}

is_testset = '📂' in input_mode

def can_run(mname):
    return mname in models

tier_pool = [m for m in TIER_MODELS[tier] if can_run(m)]

if 'Compare All' in tier:
    selected = tier_pool
else:
    st.markdown('**Select models to run:**')
    cols = st.columns(len(tier_pool))
    selected = []
    for col, mname in zip(cols, tier_pool):
        with col:
            if st.checkbox(mname, value=True, key=f'cb_{mname}'):
                selected.append(mname)

if not selected:
    st.warning('Select at least one model above to run analysis.')
    st.stop()

# ── Step 4 — Predictions ──────────────────────────────────────────────────────

st.divider()
st.subheader('Step 4 — Analysis Results')

results = []
with st.spinner('Running analysis…'):
    for mname in selected:
        if '📂' in input_mode:
            pred, conf, lat, proba = predict_from_test(
                mname, models[mname], idx, data)
        else:
            pred, conf, lat, proba = predict_from_upload(
                mname, models[mname], ecg_signal, encoders)
        if pred is not None:
            results.append(dict(
                model=mname, pred=pred, conf=conf, lat=lat, proba=proba,
                correct=(pred == true_label) if true_label else None,
                acc=MODEL_ACCURACY.get(mname, 0.0),
            ))

if not results:
    st.warning('No predictions generated.')
    st.stop()

# Consensus
vote_counts = Counter(r['pred'] for r in results)
top_vote    = vote_counts.most_common(1)[0][0]
agreement   = vote_counts[top_vote]
avg_conf    = float(np.mean([r['conf'] for r in results if r['pred'] == top_vote]))

st.markdown(f'#### Overall Assessment — {agreement} of {len(results)} models agree')
health_card(top_vote, avg_conf, agreement, len(results), true_label)

# ── Disagreement explanation ──────────────────────────────────────────────────

unique_preds = list(vote_counts.keys())
if len(unique_preds) > 1:
    # Find the model with highest validated accuracy and its prediction
    best_model  = max(results, key=lambda r: r['acc'])
    agree_label = 'most models agree' if agreement > len(results) / 2 else 'models are split'

    st.warning(
        f'**Why do models disagree?**  \n'
        f'Different models were trained with different techniques and have different '
        f'strengths. When they disagree, the most reliable result comes from the model '
        f'with the highest validated accuracy on the PTB-XL test set.  \n\n'
        f'**Best validated model:** {MODEL_FULL_NAME[best_model["model"]]} '
        f'(validated accuracy: {best_model["acc"]:.1%})  \n'
        f'**Its prediction:** {PLAIN_NAMES[best_model["pred"]]}  \n\n'
        f'Overall {agree_label} on: **{PLAIN_NAMES[top_vote]}**'
    )

# ── Per-model detail cards ────────────────────────────────────────────────────

st.markdown('---')
with st.expander(
    '📊 Individual model results — click to expand',
    expanded=(len(results) <= 4)
):
    # Validation accuracy context
    st.caption(
        'Each card shows one model\'s prediction. '
        '"Validated accuracy" is how often that model was correct across the full PTB-XL '
        'test set — higher means more reliable. The probability chart shows how confident '
        'the model is about each possible condition.'
    )
    st.markdown('')

    ncols = min(len(results), 4)
    rows  = [results[i:i + ncols] for i in range(0, len(results), ncols)]

    for row in rows:
        cols = st.columns(ncols)
        for col, r in zip(cols, row):
            sev        = SEVERITY[r['pred']]
            color      = SEVERITY_COLOR[sev]
            emoji      = SEVERITY_EMOJI[sev]
            al, ac     = acc_label(r['acc'])

            with col:
                # Model name + type
                st.markdown(
                    f'<div style="font-weight:700; font-size:0.9rem; margin-bottom:2px">'
                    f'{r["model"]}</div>'
                    f'<div style="font-size:0.75rem; color:#888; margin-bottom:6px">'
                    f'{MODEL_TYPE[r["model"]]}</div>',
                    unsafe_allow_html=True,
                )

                # Prediction badge
                st.markdown(
                    f'<div style="background:{color}18; border:1.5px solid {color}; '
                    f'border-radius:7px; padding:8px 10px; text-align:center; '
                    f'font-weight:600; color:{color}; font-size:0.85rem; margin-bottom:8px">'
                    f'{emoji}&nbsp; {PLAIN_NAMES[r["pred"]]}</div>',
                    unsafe_allow_html=True,
                )

                # Validated accuracy badge
                st.markdown(
                    f'<div style="font-size:0.78rem; margin-bottom:4px">'
                    f'Validated accuracy: '
                    f'<b style="color:{ac}">{r["acc"]:.1%} — {al}</b></div>',
                    unsafe_allow_html=True,
                )

                # Confidence + speed
                st.markdown(
                    f'<div style="font-size:0.78rem; color:#555; margin-bottom:8px">'
                    f'Prediction confidence: <b>{r["conf"]:.1%}</b> '
                    f'({conf_label(r["conf"])})<br>'
                    f'Speed: <b>{r["lat"]:.1f} ms</b></div>',
                    unsafe_allow_html=True,
                )

                # Probability bar chart
                fig2 = prob_bar(r['proba'], r['pred'])
                st.pyplot(fig2, use_container_width=False)
                plt.close()

                # Correct/incorrect note (test dataset only)
                if r['correct'] is not None:
                    note = '✅ Correct' if r['correct'] else '❌ Incorrect'
                    st.caption(f'Vs known label: {note}')

# ── Full validation metrics ───────────────────────────────────────────────────

with st.expander('📈 Research Validation Metrics — all 8 models tested', expanded=False):

    st.info(
        '**These metrics cover all 8 models evaluated in the research experiment.**  \n'
        'The live prediction interface shows only the 4 best-performing models (64–73% accuracy). '
        'The 4 feature-extraction baselines (PCA/SAE/MAE + SVM/XGBoost, 42–55%) are included '
        'here as proof of the comparative experiment but excluded from live predictions because '
        'their accuracy is too low to provide reliable guidance.  \n\n'
        'All metrics were computed on the full PTB-XL test set (4,286 recordings). '
        'The **confidence %** on each prediction card is different — that is the '
        'model\'s certainty about the specific ECG currently loaded.'
    )

    # ── Metrics table ────────────────────────────────────────────────────────
    st.markdown('#### Summary Table')

    import pandas as pd

    cv = pd.read_csv(os.path.join(DATA_DIR, 'cv_results_summary.csv'))
    cv = cv.rename(columns={
        'Model':          'Model',
        'Mode':           'Deployment Mode',
        'N_folds':        'CV Folds',
        'Mean_Accuracy':  'Accuracy',
        'Std':            'Std Dev',
        'CI_95_low':      '95% CI Low',
        'CI_95_high':     '95% CI High',
        'Mean_Latency_ms':'Avg Latency (ms)',
    })
    cv['Accuracy']      = cv['Accuracy'].map('{:.1%}'.format)
    cv['Std Dev']       = cv['Std Dev'].map('{:.3f}'.format)
    cv['95% CI Low']    = cv['95% CI Low'].map('{:.1%}'.format)
    cv['95% CI High']   = cv['95% CI High'].map('{:.1%}'.format)
    cv['Avg Latency (ms)'] = cv['Avg Latency (ms)'].map('{:.1f}'.format)
    st.dataframe(cv.set_index('Model'), use_container_width=True)

    st.caption(
        'Accuracy = proportion of correctly classified ECGs across the full test set.  '
        'Std Dev = variability across CV folds.  '
        '95% CI = confidence interval on the accuracy estimate.  '
        'Latency = average inference time per ECG.'
    )

    # ── Output images ────────────────────────────────────────────────────────
    st.markdown('---')
    st.markdown('#### F1 Score per Class — all models')
    st.caption(
        'F1 score balances precision and recall. A score of 1.0 means perfect detection; '
        '0.0 means the model never correctly identified that condition. '
        'F1 is more informative than accuracy for imbalanced datasets like PTB-XL.'
    )
    f1_path = os.path.join(DATA_DIR, 'outputs_f1_heatmap.png')
    if os.path.exists(f1_path):
        st.image(f1_path, use_container_width=True)

    st.markdown('---')
    st.markdown('#### Confusion Matrix — predicted vs actual diagnosis')
    st.caption(
        'Each row = the true (actual) diagnosis. Each column = what the model predicted. '
        'The diagonal shows correct predictions. Off-diagonal cells show where models '
        'confused one condition with another.'
    )
    conf_path = os.path.join(DATA_DIR, 'outputs_confusion.png')
    if os.path.exists(conf_path):
        st.image(conf_path, use_container_width=True)

    st.markdown('---')
    st.markdown('#### ROC Curves — detection ability per class')
    st.caption(
        'ROC (Receiver Operating Characteristic) curves show how well each model '
        'separates one condition from all others. AUC (area under curve) closer to 1.0 '
        'means better discrimination. A diagonal line (AUC = 0.5) means random guessing.'
    )
    roc_path = os.path.join(DATA_DIR, 'outputs_roc.png')
    if os.path.exists(roc_path):
        st.image(roc_path, use_container_width=True)

    st.markdown('---')
    st.markdown('#### Precision–Recall Curves')
    st.caption(
        'Precision = of all ECGs flagged as condition X, how many were actually X.  '
        'Recall = of all actual condition X cases, how many did the model catch.  '
        'High area under this curve = model is reliable even when the condition is rare.'
    )
    pr_path = os.path.join(DATA_DIR, 'outputs_precision_recall.png')
    if os.path.exists(pr_path):
        st.image(pr_path, use_container_width=True)

    st.markdown('---')
    st.markdown('#### Model Latency Comparison')
    st.caption('Average inference time per ECG for each model.')
    lat_path = os.path.join(DATA_DIR, 'outputs_latency.png')
    if os.path.exists(lat_path):
        st.image(lat_path, use_container_width=True)

# ── Condition reference ───────────────────────────────────────────────────────

with st.expander('📖 What do these heart conditions mean?'):
    for cls in CLASS_NAMES:
        sev   = SEVERITY[cls]
        color = SEVERITY_COLOR[sev]
        emoji = SEVERITY_EMOJI[sev]
        st.markdown(
            f'#### {emoji}&nbsp; {PLAIN_NAMES[cls]} '
            f'<span style="font-size:0.8rem; color:{color}">— {SEVERITY_BADGE[sev]}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(PLAIN_DESCRIPTIONS[cls])
        st.caption(f'Suggested next step: {NEXT_STEPS[cls]}')
        st.markdown('')

# ── Sidebar ───────────────────────────────────────────────────────────────────

ACTIVE_MODELS = ['SAE Fine-tuned', 'MAE Fine-tuned', 'Raw + BiLSTM', 'Raw + 1D-CNN']
RESEARCH_MODELS = ['PCA + SVM', 'SAE + SVM', 'MAE + SVM', 'MAE + XGBoost']

with st.sidebar:
    st.header('System Status')
    st.caption('**Active models (live prediction)**')
    for name in ACTIVE_MODELS:
        icon = '✅' if name in models else '❌'
        st.caption(f'{icon} {name}')
    st.divider()
    st.caption('**Research baselines (metrics only)**')
    for name in RESEARCH_MODELS:
        st.caption(f'📊 {name}')
    st.divider()
    st.caption(f'Encoders loaded: {len(encoders)} / 2')
    for key in ('sae', 'mae'):
        icon = '✅' if key in encoders else '❌'
        st.caption(f'{icon} {key.upper()} encoder')
