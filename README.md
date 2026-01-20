<h1>📊 ECG MI Classification — Raw ECG vs Standard Autoencoder vs Variational AutoEncoder</h1>
<p>
A complete deep learning project for detecting <strong>Myocardial Infarction (MI)</strong> from the
<strong>PTB-XL ECG dataset</strong> using:
</p>
<ul>
  <li>Raw ECG signals (single lead, 1000 samples)</li>
  <li>Standard Autoencoder (SAE) latent features</li>
  <li>Variational Autoencoder (VAE) latent features</li>
  <li>1D CNN classifier for MI vs Normal</li>
</ul>
<p>
This repository contains the full experiment pipeline:
</p>
<ul>
  <li>Source code</li>
  <li>Dataset download script</li>
  <li>Preprocessing and training pipeline</li>
  <li>Evaluation and visualization scripts</li>
  <li>Generated result figures</li>
  <li>README documentation</li>
</ul>

<hr />

<h2>📁 Project Structure</h2>

<pre><code>RP3_ECG_MI_Classification/
│
├── data/
│   ├── records100/                 <!-- PTB-XL waveform records (100 Hz) -->
│   ├── records500/                 <!-- PTB-XL waveform records (500 Hz) -->
│   ├── ptbxl_database.csv          <!-- PTB-XL metadata -->
│   └── scp_statements.csv          <!-- Diagnostic statement mappings -->
│
├── results/
│   ├── Figure_1.png                <!-- e.g., model comparison / accuracy -->
│   ├── Figure_2.png                <!-- confusion matrix: RAW ECG -->
│   ├── Figure_3.png                <!-- confusion matrix: AE latent -->
│   └── Figure_4.png                <!-- confusion matrix: VAE latent -->
│
├── src/
│   ├── __pycache__/                <!-- Python cache files -->
│   ├── config.py                   <!-- Global configuration & hyperparameters -->
│   ├── data_loader.py              <!-- ECG loading, filtering, preprocessing -->
│   ├── download_ptbxl.py           <!-- Script to download PTB-XL dataset -->
│   ├── main.py                     <!-- Main experiment pipeline (train & evaluate) -->
│   ├── models.py                   <!-- AE, VAE, and CNN model definitions -->
│   ├── utils.py                    <!-- Logging, metrics, helper functions -->
│   └── visualizer.py               <!-- Plotting confusion matrices & figures -->
│
├── .gitignore
├── README.md
└── requirements.txt
</code></pre>

<hr />

<h1>📘 Project Overview</h1>
<p>
This project investigates how different ECG representations affect <strong>binary MI classification</strong> (MI vs Normal).
We compare three input types:
</p>
<ul>
  <li><strong>Raw ECG</strong> (single lead, fixed length)</li>
  <li><strong>Standard Autoencoder (SAE) latent features</strong></li>
  <li><strong>Variational Autoencoder (VAE) latent features</strong></li>
</ul>
<p>
A lightweight <strong>1D CNN</strong> is trained on each representation to evaluate performance, confusion matrices, and
the impact of compression on diagnostic usefulness.
</p>
<p>
This work is part of a <strong>Machine Learning / Deep Learning coursework project</strong>.
</p>

<hr />

<h1>❓ Research Questions</h1>
<ol>
  <li>Does raw ECG outperform compressed latent features for MI detection?</li>
  <li>Can a standard autoencoder learn useful ECG representations for classification?</li>
  <li>How does a variational autoencoder (VAE) compare to a standard AE?</li>
  <li>What are the trade-offs between accuracy and computational efficiency?</li>
</ol>

<hr />

<h1>🧠 Methods &amp; Techniques</h1>

<h2>1. ECG Preprocessing</h2>
<ul>
  <li>Load PTB-XL metadata from <code>ptbxl_database.csv</code> and <code>scp_statements.csv</code></li>
  <li>Filter only <strong>Normal</strong> and <strong>MI</strong> classes</li>
  <li>Select a single ECG lead (e.g., lead 0)</li>
  <li>Resample / crop to a fixed length (e.g., 1000 samples)</li>
  <li>Apply <strong>z-score normalization</strong> per signal</li>
  <li>Split into <strong>train/test</strong> (typically 80% / 20%)</li>
</ul>

<h2>2. Standard Autoencoder (SAE)</h2>
<ul>
  <li>Encoder compresses ECG into a low-dimensional latent vector</li>
  <li>Decoder reconstructs the original ECG from the latent space</li>
  <li>Trained with <strong>MSE loss</strong> to minimize reconstruction error</li>
  <li>Latent features are later used as input to the CNN classifier</li>
</ul>

<h2>3. Variational Autoencoder (VAE)</h2>
<ul>
  <li>Encoder outputs mean and variance for latent distribution</li>
  <li>Uses the <strong>reparameterization trick</strong> to sample latent vectors</li>
  <li>Loss combines reconstruction loss + <strong>KL divergence</strong></li>
  <li>Produces smoother latent space, also fed into the CNN classifier</li>
</ul>

<h2>4. CNN Classifier</h2>
<ul>
  <li>1D convolutional layers for temporal feature extraction</li>
  <li>Max pooling and dense layers for classification</li>
  <li>Trained separately on:
    <ul>
      <li>Raw ECG signals</li>
      <li>SAE latent features</li>
      <li>VAE latent features</li>
    </ul>
  </li>
</ul>

<h2>5. Evaluation &amp; Visualization</h2>
<ul>
  <li>Accuracy and confusion matrices for each representation</li>
  <li>Comparison of <strong>TN, TP, FP, FN</strong> across models</li>
  <li>Plots generated via <code>visualizer.py</code> and saved in <code>results/</code></li>
</ul>

<hr />

<h1>📊 Results Summary</h1>

<h2>Confusion Matrix Interpretation</h2>
<ul>
  <li><strong>TN (True Negative)</strong>: Model predicted Normal, and it was actually Normal.</li>
  <li><strong>TP (True Positive)</strong>: Model predicted MI, and it was actually MI.</li>
  <li><strong>FP (False Positive)</strong>: Model predicted MI, but it was actually Normal.</li>
  <li><strong>FN (False Negative)</strong>: Model predicted Normal, but it was actually MI.</li>
</ul>

<h2>Key Findings</h2>
<ul>
  <li><strong>Raw ECG</strong> gave the best overall performance and most reliable MI detection.</li>
  <li><strong>Standard Autoencoder</strong> was faster but less accurate than raw ECG.</li>
  <li><strong>VAE latent features</strong> were smooth but showed weaker class separability.</li>
  <li>Latent-space models compressed away important MI-specific patterns, reducing diagnostic usefulness.</li>
</ul>

<hr />

<h1>🧪 How to Run the Project</h1>

<h2>1. Create and Activate a Virtual Environment (Optional but Recommended)</h2>
<pre><code>python -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows
</code></pre>

<h2>2. Install Required Libraries</h2>
<pre><code>pip install -r requirements.txt
</code></pre>

<h2>3. Download the PTB-XL Dataset</h2>
<p>
If not already present, use the provided script to download PTB-XL into the <code>data/</code> folder:
</p>
<pre><code>python src/download_ptbxl.py
</code></pre>
<p>
Ensure that the following files and folders exist in <code>data/</code>:
</p>
<ul>
  <li><code>ptbxl_database.csv</code></li>
  <li><code>scp_statements.csv</code></li>
  <li><code>records100/</code></li>
  <li><code>records500/</code></li>
</ul>

<h2>4. Run the Main Experiment Pipeline</h2>
<p>
The main script handles data loading, model training, and evaluation:
</p>
<pre><code>python src/main.py
</code></pre>
<p>
This will:
</p>
<ul>
  <li>Load and preprocess ECG data</li>
  <li>Train the Standard Autoencoder (SAE)</li>
  <li>Train the Variational Autoencoder (VAE)</li>
  <li>Train the CNN classifier on:
    <ul>
      <li>Raw ECG</li>
      <li>SAE latent features</li>
      <li>VAE latent features</li>
    </ul>
  </li>
  <li>Generate evaluation metrics and plots</li>
</ul>

<h2>5. Outputs Generated</h2>
<p>
After running <code>main.py</code>, you can find:
</p>
<ul>
  <li>Confusion matrix figures in <code>results/Figure_2.png</code>, <code>Figure_3.png</code>, <code>Figure_4.png</code></li>
  <li>Additional plots or comparisons in <code>results/Figure_1.png</code></li>
  <li>Logs and metrics (if implemented) via <code>utils.py</code> and printed to console or files</li>
</ul>

<hr />

<h1>🖼 Visual Outputs</h1>
<p>The following visual outputs are generated and stored in the <code>results/</code> folder:</p>
<ul>
  <li><strong>Figure_1.png</strong> – Overall model comparison / summary visualization (e.g., accuracy or pipeline overview).</li>
  <li><strong>Figure_2.png</strong> – Confusion matrix for the CNN trained on <strong>Raw ECG</strong>.</li>
  <li><strong>Figure_3.png</strong> – Confusion matrix for the CNN trained on <strong>AE latent features</strong>.</li>
  <li><strong>Figure_4.png</strong> – Confusion matrix for the CNN trained on <strong>VAE latent features</strong>.</li>
</ul>

<hr />

<h1>📄 Conclusion &amp; Future Work</h1>
<h2>Conclusion</h2>
<ul>
  <li>Raw ECG gave the best overall performance.</li>
  <li>Standard Autoencoder saved time but achieved lower accuracy than raw ECG.</li>
  <li>Both latent-feature models struggled to detect MI cases reliably.</li>
  <li>Raw ECG signals preserved full morphology, giving clearly superior classification performance.</li>
</ul>

<h2>Future Work</h2>
<ul>
  <li>Explore deeper or multi-lead CNN architectures for richer MI feature extraction.</li>
  <li>Increase latent-space capacity or use supervised autoencoders to preserve MI-specific patterns.</li>
  <li>Experiment with longer training schedules and data augmentation to reduce false negatives.</li>
  <li>Integrate clinical metadata and multi-label outputs for more realistic diagnostic modeling.</li>
</ul>

<hr />

<h1>👨‍💻 Author</h1>
<p>
<strong>Mohammed Irshad</strong><br />
Vytautas Magnus University<br />
ECG MI Classification – Deep Learning Project
</p>

<hr />

<h1>📚 References</h1>
<ul>
  <li>PTB-XL: A large publicly available electrocardiography dataset – PhysioNet.</li>
  <li>Kingma, D. P., &amp; Welling, M. – Auto-Encoding Variational Bayes.</li>
  <li>Goodfellow, I. et al. – Deep Learning, MIT Press.</li>
  <li>Relevant ECG classification and autoencoder-based representation learning literature.</li>
</ul>

<hr />
