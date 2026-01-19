import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE

class ResultVisualizer:
    def __init__(self, results_dict, y_test):
        """
        results_dict: dictionary containing 'y_pred', 'y_probs', and 'accuracy' for each pipeline.
        y_test: true labels.
        """
        self.results = results_dict
        self.y_test = y_test
        sns.set_theme(style="whitegrid")

    def plot_accuracy_comparison(self):
        plt.figure(figsize=(10, 6))
        names = list(self.results.keys())
        accs = [self.results[n]['accuracy'] for n in names]
        
        bars = plt.bar(names, accs, color=sns.color_palette("viridis", len(names)))
        plt.title("Classification Performance (Lead I ECG)", fontsize=15, pad=20)
        plt.ylabel("Accuracy Score", fontsize=12)
        plt.ylim(0, 1.1)
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", 
                     ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show() 

    def plot_confusion_matrices(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, (name, data) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, data['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues', cbar=False)
            axes[i].set_title(f"CM: {name}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")
        
        plt.tight_layout()
        plt.show()  

    def plot_roc_curves(self):
        plt.figure(figsize=(8, 6))
        for name, data in self.results.items():
            if np.isnan(data['y_probs']).any():
                print(f"[WARNING] Skipping ROC for {name} due to NaN values")
                continue

            fpr, tpr, _ = roc_curve(self.y_test, data['y_probs'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
            
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()  

    def plot_latent_tsne(self, latent_features, labels, title="t-SNE"):
        if np.isnan(latent_features).any():
            print(f"[WARNING] Skipping {title} because data contains NaNs.")
            return

        plt.figure(figsize=(10, 8))
        tsne = TSNE(n_components=2, random_state=42)
        projections = tsne.fit_transform(latent_features)
        
        plt.scatter(projections[:, 0], projections[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
        plt.colorbar(label='0: Normal, 1: MI')
        plt.title(title)
        plt.show()  

    def plot_reconstructions(self, original, ae, vae, n=3):
        plt.figure(figsize=(15, 10))
        for i in range(n):
            # Original
            plt.subplot(3, n, i + 1)
            plt.plot(original[i])
            if i == 0:
                plt.ylabel("Original")
            
            # Standard Autoencoder Reconstruction
            plt.subplot(3, n, i + 1 + n)
            recon_ae = ae.predict(original[i:i+1])
            plt.plot(recon_ae[0])
            if i == 0:
                plt.ylabel("SAE Recon")

            # VAE Reconstruction
            plt.subplot(3, n, i + 1 + 2*n)
            recon_vae = vae.predict(original[i:i+1])
            if not np.isnan(recon_vae).any():
                plt.plot(recon_vae[0])
            else:
                plt.text(0.5, 0.5, "NaN Error", ha='center')
            if i == 0:
                plt.ylabel("VAE Recon")
            
        plt.tight_layout()
        plt.show() 
