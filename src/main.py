import sys, os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Fix MKL primitive error on Windows
import time
import numpy as np
import psutil
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from utils import log
from data_loader import ECGDataLoader
from models import ModelBuilder
from visualizer import ResultVisualizer

def apply_mae_mask(X, patch_size=config.PATCH_SIZE, mask_ratio=config.MASK_RATIO):
    """Zeroes out patches of the signal to force the network to learn the ECG"""
    X_masked = X.copy()
    num_patches = X.shape[1] // patch_size
    num_mask = int(num_patches * mask_ratio)
    for i in range(X.shape[0]):
        mask_indices = np.random.choice(num_patches, num_mask, replace=False)
        for idx in mask_indices:
            start = idx * patch_size
            end = start + patch_size
            X_masked[i, start:end, :] = 0
    return X_masked

def timed_predict(model, X, name):
    """Returns predictions and inference time in seconds"""
    start = time.time()
    preds = model.predict(X, verbose=0)
    elapsed = time.time() - start
    log(f"Inference [{name}]: {elapsed*1000:.1f}ms for {len(X)} samples")
    return preds, elapsed

def timed_predict_ml(clf, X, name):
    """Returns predictions and inference time for sklearn models"""
    start = time.time()
    preds = clf.predict(X)
    elapsed = time.time() - start
    log(f"Inference [{name}]: {elapsed*1000:.1f}ms for {len(X)} samples")
    return preds, elapsed

def run_pipeline():
    process = psutil.Process(os.getpid())

    log("1. Loading and Preprocessing 12-Lead Data...")
    loader = ECGDataLoader().load_and_process()
    builder = ModelBuilder()
    research_data = {}
    latency_data = {}
    proba_data = {}  # for AUC-ROC curves

    log("2. Generating MAE Masked Data...")
    X_train_masked = apply_mae_mask(loader.X_train)

    log("3. Training Feature Extractors...")
    sae, encoder_sae = builder.build_autoencoder()
    log("Training SAE...")
    sae.fit(loader.X_train, loader.X_train, epochs=config.EPOCHS_AE,
            batch_size=config.BATCH_SIZE, verbose=0)

    mae, encoder_mae = builder.build_autoencoder()
    log("Training MAE (Self-Supervised)...")
    mae.fit(X_train_masked, loader.X_train, epochs=config.EPOCHS_AE,
            batch_size=config.BATCH_SIZE, verbose=0)

    # Extract Latent Features
    X_train_sae = encoder_sae.predict(loader.X_train, verbose=0)
    X_test_sae = encoder_sae.predict(loader.X_test, verbose=0)
    X_train_mae = encoder_mae.predict(loader.X_train, verbose=0)
    X_test_mae = encoder_mae.predict(loader.X_test, verbose=0)

    # PCA Baseline
    log("Fitting PCA Baseline...")
    pca = PCA(n_components=config.LATENT_DIM)
    X_train_flat = loader.X_train.reshape(loader.X_train.shape[0], -1)
    X_test_flat = loader.X_test.reshape(loader.X_test.shape[0], -1)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)

    log("4. Training Classifiers...")

    # Mode A: Conventional ML
    ml_pipelines = [
        ("PCA + SVM",       X_train_pca, X_test_pca, SVC(probability=True)),
        ("SAE + SVM",       X_train_sae, X_test_sae, SVC(probability=True)),
        ("MAE + SVM (Ours)",X_train_mae, X_test_mae, SVC(probability=True)),
        ("MAE + XGBoost",   X_train_mae, X_test_mae, XGBClassifier(eval_metric='mlogloss')),
    ]

    for name, x_tr, x_te, clf in ml_pipelines:
        log(f"Training {name}...")
        clf.fit(x_tr, loader.y_train)
        preds, elapsed = timed_predict_ml(clf, x_te, name)
        proba = clf.predict_proba(x_te)
        research_data[name] = {"accuracy": accuracy_score(loader.y_test, preds), "y_pred": preds}
        latency_data[name] = elapsed
        proba_data[name] = proba

    # Mode B: Deep Learning — BiLSTM
    log("Training Raw + BiLSTM...")
    bilstm = builder.build_bilstm()
    bilstm.fit(loader.X_train, loader.y_train, epochs=config.EPOCHS_DL,
               batch_size=config.BATCH_SIZE, verbose=0)
    bilstm_probs, elapsed = timed_predict(bilstm, loader.X_test, "BiLSTM")
    bilstm_preds = np.argmax(bilstm_probs, axis=1)
    research_data["Raw + BiLSTM"] = {"accuracy": accuracy_score(loader.y_test, bilstm_preds), "y_pred": bilstm_preds}
    latency_data["Raw + BiLSTM"] = elapsed
    proba_data["Raw + BiLSTM"] = bilstm_probs

    # Mode B: Deep Learning — 1D-CNN
    log("Training Raw + 1D-CNN...")
    cnn = builder.build_cnn()
    cnn.fit(loader.X_train, loader.y_train, epochs=config.EPOCHS_DL,
            batch_size=config.BATCH_SIZE, verbose=0)
    cnn_probs, elapsed = timed_predict(cnn, loader.X_test, "1D-CNN")
    cnn_preds = np.argmax(cnn_probs, axis=1)
    research_data["Raw + 1D-CNN"] = {"accuracy": accuracy_score(loader.y_test, cnn_preds), "y_pred": cnn_preds}
    latency_data["Raw + 1D-CNN"] = elapsed
    proba_data["Raw + 1D-CNN"] = cnn_probs

    ram_used = process.memory_info().rss / 1024 / 1024
    log(f"Peak RAM usage: {ram_used:.0f} MB")

    log("5. Generating Master Thesis Visuals...")
    viz = ResultVisualizer(research_data, loader.y_test)

    # Task 2: MAE masking strategy
    viz.plot_mae_masking(loader.X_train[0], X_train_masked[0])

    # Task 3: MAE reconstruction quality
    viz.plot_reconstruction(loader.X_train[0], X_train_masked[0], mae)

    # Task 4/5: Accuracy comparison
    viz.plot_accuracy_comparison()

    # Task 5: Per-class F1 heatmap
    viz.plot_f1_heatmap()

    # Task 5: Inference latency benchmarking
    viz.plot_resource_metrics(latency_data)

    # Task 5: Latent space t-SNE
    viz.plot_tsne(X_test_mae, X_test_sae)

    # Task 6: XAI Saliency Maps (using CNN — best fit for gradient saliency)
    log("Generating XAI Saliency Maps...")
    sample_idx = 0
    viz.plot_saliency_map(cnn, loader.X_test[sample_idx],
                          loader.y_test[sample_idx], model_name="1D-CNN")

    # Task 5: Precision & Recall
    viz.plot_precision_recall_table()

    # Task 5: AUC-ROC curves
    viz.plot_roc_curves(proba_data)

    # Task 4/5: Confusion matrices
    viz.plot_confusion_matrices()

    log("Complete. All thesis visuals saved!")
    log(f"Outputs: accuracy, f1_heatmap, latency, tsne, saliency, reconstruction, masking, confusion")

if __name__ == "__main__":
    run_pipeline()
