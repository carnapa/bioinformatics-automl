# Bioinformatics AutoML Pipeline & Dashboard

A config-driven machine learning pipeline and interactive web dashboard for bioinformatics datasets. Point it at a CSV/Excel file or use the interactive web UI to get end-to-end results: exploratory data analysis, preprocessing, model training with hyperparameter tuning, RNA-Seq differential expression analysis, and PyTorch deep learning on gene expression data.

## Features

### Classical AutoML
- **Interactive Shiny Web Dashboard** (`app.py`) — reactive GUI powered by Shiny for Python with real-time data preview, dynamic EDA visualisations, non-blocking model training, interactive leaderboards, and artifact downloads.
- **Serverless WebAssembly Deployment** (`docs/`) — fully static Shinylive build that runs entirely in the client's browser on GitHub Pages with zero server setup.
- **Headless CLI Pipeline** (`main.py`) — automated batch execution controlled from a declarative `config.yaml`.
- **Automated EDA** — statistical summaries, correlation heatmaps, class-balance plots, PCA with configurable components.
- **Preprocessing** — missing-value imputation (mean/median/zero), log transform, feature scaling (standard/minmax/robust).
- **Auto Task Detection** — automatically identifies classification vs. regression based on target column characteristics.
- **Model Training** — hyperparameter tuning via `GridSearchCV` with stratified cross-validation (Classification: Random Forest, Logistic Regression, SVM, KNN | Regression: Random Forest, Ridge, SVR, KNN).
- **Evaluation & Persistence** — classification reports, confusion matrices, ROC curves, feature importance plots, comparison CSV, and `.joblib` model serialisation.
- **Export & Downloads** — download individual `.joblib` model artifacts, leaderboard CSVs, or a complete bundled `.zip` archive.

### RNA-Seq Transcriptomics (new)
- **`RNASeqPipeline`** (`src/bio/rnaseq.py`) — end-to-end RNA-Seq processing in Python:
  - CPM normalisation with configurable minimum-expression gene filtering.
  - `log2(CPM + 1)` transformation.
  - Differential Expression Analysis (DEA) via Welch's t-test or Mann-Whitney U, with Benjamini-Hochberg FDR correction.
  - Volcano plot and MA plot (return `matplotlib.Figure` objects; fully headless).
  - Pathway enrichment via [gseapy](https://gseapy.readthedocs.io/) (KEGG, Reactome, GO, etc.) with an automatic offline fallback.
- **Tab 4 — "Transcriptomics & DEA"** in the Shiny dashboard: upload count matrix + metadata CSVs, select groups, adjust Log2FC / FDR thresholds, and get interactive volcano plot, MA plot, and downloadable DEA results table.

### PyTorch Deep Learning on Gene Expression (new)
- **`ExpressionMLP`** (`src/models/dl_expression.py`) — flexible multi-layer perceptron for classification or regression on high-dimensional expression matrices (`Linear → BatchNorm1d → ReLU → Dropout` per layer).
- **`ExpressionTrainer`** — training loop with Adam optimiser, early stopping, best-weight restoration, and training-curve plots.
- **`ExpressionVAE`** — Variational Autoencoder for non-linear dimensionality reduction; encoder outputs `(mu, logvar)`, deterministic `transform()` returns `mu`.
- **`VAELoss`** — MSE reconstruction + beta-weighted KL divergence (supports beta-VAE annealing).
- **`VAETrainer`** — dedicated VAE training loop with linear KL warmup schedule and 3-panel loss curve figure.
- **Tab 5 — "PyTorch Expression DL"** in the Shiny dashboard: select MLP or VAE, configure architecture and hyperparameters, train non-blocking, inspect loss curves / latent-space PCA scatter, download `.pt` checkpoint.

---

## Project Structure

```
bioinformatics-automl/
├── app.py                      # Interactive Shiny Express Web App (5 tabs)
├── main.py                     # Headless CLI pipeline entry point
├── config.yaml                 # All pipeline settings
├── pytest.ini                  # Pytest configuration (marker definitions)
├── requirements.txt            # Python dependencies
├── environment.yaml            # Conda environment spec
├── data/                       # Input datasets
├── results/                    # Generated outputs (plots, reports, models)
├── docs/                       # Static Shinylive WebAssembly build (GitHub Pages)
├── src/
│   ├── utils.py                # Config & data loading utilities
│   ├── eda.py                  # Exploratory data analysis module
│   ├── automl_models.py        # Sklearn model training, tuning & evaluation
│   ├── bio/
│   │   ├── __init__.py
│   │   └── rnaseq.py           # RNASeqPipeline — normalisation, DEA, plots, enrichment
│   └── models/
│       ├── __init__.py
│       └── dl_expression.py    # ExpressionMLP, ExpressionVAE, Trainers, VAELoss
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Shared pytest fixtures (count matrix, metadata, log-CPM)
│   └── test_rnaseq_dl.py       # 30-test suite for RNA-Seq & DL modules
└── notebooks/                  # Jupyter notebooks (exploratory work)
```

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/carnapa/bioinformatics-automl.git
cd bioinformatics-automl

# Install dependencies
pip install -r requirements.txt
```

Or use the Conda environment:

```bash
conda env create -f environment.yaml
conda activate bioinformatics-automl
```

### 2. Option A: Launch Interactive Web UI (Local Server)

```bash
shiny run --reload app.py
```

Open **`http://127.0.0.1:8000`** in your browser. The dashboard has five tabs:

| Tab | Description |
|-----|-------------|
| **Data Preview & Stats** | Upload dataset, preview, summary statistics |
| **Exploratory Data Analysis** | Correlation heatmap, PCA, target distribution |
| **Models & Leaderboard** | Train sklearn models, compare results, download artifacts |
| **Transcriptomics & DEA** | Upload RNA-Seq counts + metadata, run DEA, download results |
| **PyTorch Expression DL** | Train MLP or VAE on normalised expression matrix |

### 3. Option B: Run Serverless WebAssembly Version (No Python Server)

The app is exported to WebAssembly in `docs/` using Shinylive:

```bash
python -m http.server --directory docs --bind localhost 8008
```

Open **`http://localhost:8008`**, or deploy to **GitHub Pages** (Settings → Pages → Directory: `/docs`).

> **Note:** The RNA-Seq and PyTorch tabs require server-side Python and are not available in the Shinylive WebAssembly build.

### 4. Option C: Run Headless CLI Pipeline

```bash
# Edit config.yaml to point to your dataset, then run:
python main.py
```

Results (plots, reports, saved models) are written to the `results/` directory.

To enable the RNA-Seq or deep learning pipelines in headless mode, set `enabled: true` in the corresponding `config.yaml` sections and provide paths to your count matrix and metadata files.

---

## Configuration

All settings for the headless CLI pipeline live in [`config.yaml`](config.yaml):

| Section | What it controls |
|---------|-----------------|
| `data` | Input file path, target column, index column, columns to drop |
| `preprocessing` | Imputation strategy, scaling method, log transform, test split size |
| `eda` | Correlation threshold, PCA components, top-N features/categories |
| `models` | List of models to train |
| `model_training` | Task type (auto/classification/regression), CV folds, scoring metric, tuning toggle, custom hyperparameter grids |
| `output` | Results directory, save plots/models toggles |
| `rnaseq` | Count matrix path, metadata path, CPM filter, DEA groups, FDR/Log2FC thresholds, enrichment database |
| `deep_learning` | Model type (mlp/vae), hidden dims, latent dim, epochs, learning rate, dropout, batch size |

---

## Supported Models

### Classical AutoML (scikit-learn)

| Classification | Regression |
|----------------|------------|
| Random Forest | Random Forest |
| Logistic Regression | Ridge Regression |
| SVM (SVC) | SVR |
| K-Nearest Neighbors | K-Nearest Neighbors |

### Deep Learning (PyTorch)

| Model | Use Case |
|-------|----------|
| `ExpressionMLP` | Classification or regression on expression matrices |
| `ExpressionVAE` | Non-linear dimensionality reduction / latent-space exploration |

---

## Running Tests

```bash
# Fast tests only (excludes training loops) — ~2 seconds
pytest tests/test_rnaseq_dl.py -v -m "not slow"

# Full test suite including training-loop smoke tests — ~5 seconds
pytest tests/test_rnaseq_dl.py -v
```

The test suite covers:
- CPM normalisation shape and value correctness
- DEA output schema, p-value ranges, `significant` dtype
- Volcano and MA plot return types (and absence of `plt.show()`)
- `ExpressionDataset` length and dtype
- `ExpressionMLP` forward-pass output shape
- `ExpressionVAE` forward-pass shapes, determinism in eval mode, stochasticity in train mode
- `VAELoss` positivity and beta scaling
- `VAETrainer.transform()` determinism and output shape
- `@pytest.mark.slow`: MLP and VAE training-loop smoke tests (fit, evaluate, plot)

---

## Requirements

- Python 3.9+
- **Core:** pandas, scikit-learn, seaborn, matplotlib, pyyaml, joblib, shiny, shinyswatch
- **RNA-Seq:** numpy, scipy, statsmodels, gseapy
- **Deep Learning:** torch (CPU or CUDA)

See [`requirements.txt`](requirements.txt) for pinned versions and [`environment.yaml`](environment.yaml) for the full Conda spec.

---

## License

[MIT](LICENSE)
