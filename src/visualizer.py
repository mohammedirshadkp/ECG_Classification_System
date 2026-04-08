import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import config

class ResultVisualizer:
    def __init__(self, results_dict, y_test):
        self.results = results_dict
        self.y_test = y_test
        self.class_names = list(config.LABEL_MAP.keys())
        sns.set_theme(style="whitegrid")

    def plot_mae_masking(self, original, masked):
        """Side-by-side: original vs masked"""
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        ax1.plot(original[:, 0], color='steelblue', linewidth=0.8)
        ax1.set_title("Original ECG Signal (Lead I)", fontweight='bold')
        ax1.set_ylabel("Amplitude")

        ax2.plot(masked[:, 0], color='crimson', linewidth=0.8)
        ax2.set_title(f"Masked Input ({int(config.MASK_RATIO*100)}% hidden — MAE must reconstruct original)", fontweight='bold')
        ax2.set_ylabel("Amplitude")
        ax2.set_xlabel("Samples (time steps)")

        plt.suptitle("MAE Self-Supervised Training Strategy", fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig("outputs_mae_masking.png", bbox_inches='tight', dpi=150)
        plt.show()

    def plot_reconstruction(self, original, masked, mae_model):
        """Original vs Masked vs MAE Reconstruction — key thesis slide"""
        reconstructed = mae_model.predict(masked[np.newaxis, ...], verbose=0)[0]

        _, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(original[:, 0], color='steelblue', linewidth=0.8)
        axes[0].set_title("1. Original ECG (Ground Truth)", fontweight='bold')
        axes[0].set_ylabel("Amplitude")

        axes[1].plot(masked[:, 0], color='crimson', linewidth=0.8)
        axes[1].set_title("2. Masked Input (75% hidden — what the MAE sees)", fontweight='bold')
        axes[1].set_ylabel("Amplitude")

        axes[2].plot(reconstructed[:, 0], color='forestgreen', linewidth=0.8)
        axes[2].set_title("3. MAE Reconstruction (what the model learned to predict)", fontweight='bold')
        axes[2].set_ylabel("Amplitude")
        axes[2].set_xlabel("Samples (time steps)")

        plt.suptitle("MAE Reconstruction Quality: Learning ECG Structure from Masked Input",
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig("outputs_reconstruction.png", bbox_inches='tight', dpi=150)
        plt.show()

    def plot_saliency_map(self, model, X_sample, true_label, model_name="CNN"):
        """XAI: highlights which ECG segments most influenced the classification"""
        x_tensor = tf.convert_to_tensor(X_sample[np.newaxis, ...], dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            pred = model(x_tensor, training=False)
            predicted_class = int(tf.argmax(pred[0]))
            class_score = pred[0, predicted_class]

        grads = tape.gradient(class_score, x_tensor)[0].numpy()
        saliency = np.mean(np.abs(grads), axis=1)  # avg over 12 leads -> (1000,)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        ax1.plot(X_sample[:, 0], color='steelblue', linewidth=0.8, label='ECG Lead I')
        ax1.set_title(f"XAI Saliency Map — {model_name}\nTrue: {self.class_names[true_label]}  |  Predicted: {self.class_names[predicted_class]}",
                      fontweight='bold')
        ax1.set_ylabel("Amplitude")
        ax1.legend()

        # Heatmap below signal
        ax2.imshow(saliency[np.newaxis, :], aspect='auto', cmap='hot', extent=[0, len(saliency), 0, 1])
        ax2.set_title("Saliency Intensity (red = high influence on classification)", fontsize=9)
        ax2.set_ylabel("Importance")
        ax2.set_xlabel("Samples (time steps)")

        plt.suptitle("Explainability (XAI): What the Model Looks at in the ECG",
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig("outputs_saliency.png", bbox_inches='tight', dpi=150)
        plt.show()

    def plot_resource_metrics(self, latency_dict):
        """Benchmarking: inference time per model"""
        names = list(latency_dict.keys())
        times = [latency_dict[n] * 1000 for n in names]  # convert to ms

        _, ax = plt.subplots(figsize=(12, 5))
        colors = ['#e63946' if 'MAE' in n else '#457b9d' for n in names]
        bars = ax.bar(names, times, color=colors, edgecolor='white', linewidth=1.2)

        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{t:.1f}ms", ha='center', fontweight='bold', fontsize=10)

        ax.set_title("Task 5 — Inference Latency per Model (lower = more efficient)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Inference Time (ms)")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha='right')

        ml_patch = mpatches.Patch(color='#457b9d', label='Conventional ML / DL')
        mae_patch = mpatches.Patch(color='#e63946', label='MAE-based (Ours)')
        ax.legend(handles=[ml_patch, mae_patch])

        plt.tight_layout()
        plt.savefig("outputs_latency.png", bbox_inches='tight', dpi=150)
        plt.show()

    def plot_accuracy_comparison(self):
        """Bar chart with MAE highlighted"""
        plt.figure(figsize=(12, 6))
        names = list(self.results.keys())
        accs = [self.results[n]['accuracy'] for n in names]
        colors = ['#e63946' if 'Ours' in n else '#2d004b' for n in names]

        bars = plt.bar(names, accs, color=colors, edgecolor='white', linewidth=1.2)
        plt.title("ECG Classification Accuracy: All Models Compared", fontsize=14, fontweight='bold')
        plt.ylabel("Accuracy Score")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=15, ha='right')

        for bar, acc in zip(bars, accs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                     f"{acc:.1%}", ha='center', fontweight='bold', fontsize=11)

        plt.tight_layout()
        plt.savefig("outputs_accuracy.png", bbox_inches='tight', dpi=150)
        plt.show()

    def plot_f1_heatmap(self):
        """Per-class F1 score heatmap"""
        f1_matrix = []
        for _, data in self.results.items():
            f1 = f1_score(self.y_test, data['y_pred'], average=None,
                          labels=list(range(len(self.class_names))), zero_division=0)
            f1_matrix.append(f1)

        model_names = list(self.results.keys())
        plt.figure(figsize=(10, 6))
        sns.heatmap(np.array(f1_matrix), annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=self.class_names, yticklabels=model_names,
                    vmin=0, vmax=1, linewidths=0.5, linecolor='white',
                    annot_kws={"size": 11, "weight": "bold"})
        plt.title("Per-Class F1 Score: Which Model Detects Which Condition Best",
                  fontsize=13, fontweight='bold')
        plt.xlabel("Cardiac Condition", fontsize=11)
        plt.ylabel("Model", fontsize=11)
        plt.tight_layout()
        plt.savefig("outputs_f1_heatmap.png", bbox_inches='tight', dpi=150)
        plt.show()

    def plot_tsne(self, X_mae_features, X_sae_features):
        """t-SNE latent space: SAE vs MAE side by side"""
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        colors = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261']

        for features, ax, title in [
            (X_sae_features, ax1, "SAE Latent Space"),
            (X_mae_features, ax2, "MAE Latent Space (Ours)")
        ]:
            tsne = TSNE(n_components=2, random_state=config.RANDOM_STATE, perplexity=30, max_iter=500)
            reduced = tsne.fit_transform(features)
            for i, cls in enumerate(self.class_names):
                mask = self.y_test == i
                ax.scatter(reduced[mask, 0], reduced[mask, 1],
                           c=colors[i], label=cls, alpha=0.6, s=20)
            ax.set_title(title, fontweight='bold', fontsize=12)
            ax.legend(loc='upper right', markerscale=2)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")

        plt.suptitle("Latent Space Visualization: Can the model separate cardiac conditions?",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig("outputs_tsne.png", bbox_inches='tight', dpi=150)
        plt.show()

    def plot_confusion_matrices(self):
        """Larger readable confusion matrices in 2x3 grid"""
        n = len(self.results)
        _, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, (name, data) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, data['y_pred'])
            acc = data['accuracy']
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues', cbar=False,
                        xticklabels=self.class_names, yticklabels=self.class_names,
                        annot_kws={"size": 10})
            axes[i].set_title(f"{name}\nAcc: {acc:.1%}", fontweight='bold', fontsize=11)
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Confusion Matrices: All Models", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("outputs_confusion.png", bbox_inches='tight', dpi=150)
        plt.show()

    def plot_precision_recall_table(self):
        """Precision and Recall per model — printed as a heatmap table"""
        n_classes = len(self.class_names)
        model_names = list(self.results.keys())

        precision_matrix = []
        recall_matrix = []
        for _, data in self.results.items():
            p = precision_score(self.y_test, data['y_pred'], average=None,
                                labels=list(range(n_classes)), zero_division=0)
            r = recall_score(self.y_test, data['y_pred'], average=None,
                             labels=list(range(n_classes)), zero_division=0)
            precision_matrix.append(p)
            recall_matrix.append(r)

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        sns.heatmap(np.array(precision_matrix), annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=model_names,
                    vmin=0, vmax=1, linewidths=0.5, linecolor='white',
                    annot_kws={"size": 11, "weight": "bold"}, ax=ax1)
        ax1.set_title("Precision per Class", fontweight='bold', fontsize=12)
        ax1.set_xlabel("Cardiac Condition")
        ax1.set_ylabel("Model")

        sns.heatmap(np.array(recall_matrix), annot=True, fmt='.2f', cmap='Greens',
                    xticklabels=self.class_names, yticklabels=model_names,
                    vmin=0, vmax=1, linewidths=0.5, linecolor='white',
                    annot_kws={"size": 11, "weight": "bold"}, ax=ax2)
        ax2.set_title("Recall per Class", fontweight='bold', fontsize=12)
        ax2.set_xlabel("Cardiac Condition")
        ax2.set_ylabel("Model")

        plt.suptitle("Precision & Recall: How Reliable and Complete is Each Model's Detection?",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig("outputs_precision_recall.png", bbox_inches='tight', dpi=150)
        plt.show()

    def plot_roc_curves(self, proba_dict):
        """AUC-ROC curves — one subplot per model, all 5 classes overlaid"""
        n_models = len(proba_dict)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        _, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
        axes = axes.flatten()

        y_bin = label_binarize(self.y_test, classes=list(range(len(self.class_names))))
        colors = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261']

        for i, (name, proba) in enumerate(proba_dict.items()):
            ax = axes[i]
            for j, cls in enumerate(self.class_names):
                fpr, tpr, _ = roc_curve(y_bin[:, j], proba[:, j])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[j], linewidth=1.5,
                        label=f"{cls} (AUC={roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)
            ax.set_title(name, fontweight='bold', fontsize=11)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc='lower right', fontsize=8)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])

        for j in range(n_models, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("AUC-ROC Curves: Diagnostic Performance per Cardiac Condition",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("outputs_roc.png", bbox_inches='tight', dpi=150)
        plt.show()
