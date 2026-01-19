import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils import log
from data_loader import ECGDataLoader
from models import ModelBuilder
from visualizer import ResultVisualizer

def run_pipeline():
    # 1. Load and Preprocess Data
  
    loader = ECGDataLoader().load_and_process()
    builder = ModelBuilder()
    research_data = {}

    # 2. Autoencoder (AE) Pipeline
   
    log("Running AE Pipeline...")
    ae, encoder_ae = builder.build_ae()
    # verbose=2 prints one line per epoch
    ae.fit(loader.X_train_flat, loader.X_train_flat, 
           epochs=config.EPOCHS_AE, 
           batch_size=config.BATCH_SIZE, 
           verbose=2)
    
    z_ae_train = encoder_ae.predict(loader.X_train_flat)
    z_ae_test = encoder_ae.predict(loader.X_test_flat)

    # 3. Variational Autoencoder (VAE) Pipeline
    
    log("Running VAE Pipeline...")
    vae, encoder_vae = builder.build_vae()
    vae.fit(loader.X_train_flat, 
           epochs=config.EPOCHS_VAE, 
           batch_size=config.BATCH_SIZE, 
           verbose=2)
    
    z_vae_train, _, _ = encoder_vae.predict(loader.X_train_flat)
    z_vae_test, _, _ = encoder_vae.predict(loader.X_test_flat)

    # 4. Classification & Evaluation

    # We compare three different inputs for the CNN classifier
    pipelines = [
        ("RAW ECG", loader.X_train_cnn, loader.X_test_cnn, config.MAX_SAMPLES),
        ("AE Latent", z_ae_train[..., np.newaxis], z_ae_test[..., np.newaxis], config.LATENT_DIM),
        ("VAE Latent", z_vae_train[..., np.newaxis], z_vae_test[..., np.newaxis], config.LATENT_DIM)
    ]

    for name, train_data, test_data, length in pipelines:
        log(f"Training CNN on {name}")
        cnn = builder.build_cnn(length)
        cnn.fit(train_data, loader.y_train, 
                epochs=config.EPOCHS_CNN, 
                batch_size=config.BATCH_SIZE, 
                verbose=2)
        
        probs = cnn.predict(test_data).flatten()
        preds = (probs > 0.5).astype(int)
        
        research_data[name] = {
            'accuracy': accuracy_score(loader.y_test, preds),
            'y_pred': preds,
            'y_probs': probs
        }

    # 5. Generate Research Figures
    log("Generating Research-Grade Visualizations...")
    viz = ResultVisualizer(research_data, loader.y_test)
    
    # Standard Performance Metrics
    viz.plot_accuracy_comparison()
    viz.plot_confusion_matrices()
    viz.plot_roc_curves()
    
    # Advanced Research Insights
    log("Generating t-SNE and Reconstruction plots...")
    viz.plot_latent_tsne(z_vae_test, loader.y_test, title="VAE Latent Space t-SNE")
    viz.plot_reconstructions(loader.X_test_flat, ae, vae, n=3)
    
    log("Pipeline complete. All figures saved as .png files in the src folder.")

if __name__ == "__main__":
    run_pipeline()