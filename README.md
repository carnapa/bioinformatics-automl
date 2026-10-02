# Bioinformatics AutoML Pipeline & Dashboard

A config-driven machine learning pipeline and interactive web dashboard for bioinformatics datasets. Point it at a CSV/Excel file or use the interactive web UI to get end-to-end results: exploratory data analysis, preprocessing, model training with hyperparameter tuning, and evaluation reports with visualisations.

## Features

- **Interactive Shiny Web Dashboard** (`app.py`) — reactive GUI powered by Shiny for Python with real-time data preview, dynamic EDA visualizations, non-blocking model training, interactive leaderboards, and artifact downloads.
- **Serverless WebAssembly Deployment** (`docs/`) — fully static Shinylive build that runs entirely in the client's web browser on GitHub Pages with zero server setup.
- **Headless CLI Pipeline** (`main.py`) — automated batch execution controlled from a declarative `config.yaml`.
- **Automated EDA** — statistical summaries, correlation heatmaps, class-balance plots, PCA with configurable components.
- **Preprocessing** — missing-value imputation (mean/median/zero), log transform, feature scaling (standard/minmax/robust).
- **Auto Task Detection** — automatically identifies classification vs. regression based on target column characteristics.
- **Model Training** — hyperparameter tuning via `GridSearchCV` with stratified cross-validation (Classification: Random Forest, Logistic Regression, SVM, KNN | Regression: Random Forest, Ridge, SVR, KNN).
- **Evaluation & Persistence** — classification reports, confusion matrices, ROC curves, feature importance plots, comparison CSV, and `.joblib` model serialization.
- **Export & Downloads** — download individual `.joblib` model artifacts, leaderboard CSVs, or a complete bundled `.zip` archive.

---

## Project Structure

```
bioinformatics-automl/
├── app.py               # Interactive Shiny Express Web App
├── main.py              # Headless CLI pipeline entry point
├── config.yaml          # All pipeline settings
├── requirements.txt     # Python dependencies
├── data/                # Input datasets
├── results/             # Generated outputs (plots, reports, models)
├── docs/                # Static Shinylive WebAssembly build (GitHub Pages)
├── src/
│   ├── utils.py         # Config & data loading utilities
│   ├── eda.py           # Exploratory data analysis module
│   └── models.py        # Model training, tuning & evaluation
└── notebooks/           # Jupyter notebooks (exploratory work)
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

### 2. Option A: Launch Interactive Web UI (Local Server)

```bash
shiny run --reload app.py
```
Open **`http://127.0.0.1:8000`** in your browser to upload data, run EDA, train models, and inspect interactive visual reports.

### 3. Option B: Run Serverless WebAssembly Version (No Python Server)

The app is exported to WebAssembly in `docs/` using Shinylive:
```bash
python -m http.server --directory docs --bind localhost 8008
```
Open **`http://localhost:8008`** to test the client-side Wasm app, or deploy to **GitHub Pages** (Settings $\rightarrow$ Pages $\rightarrow$ Directory: `/docs`).

### 4. Option C: Run Headless CLI Pipeline

```bash
# Edit config.yaml to point to your dataset, then run:
python main.py
```
Results (plots, reports, saved models) are written to the `results/` directory.

---

## Configuration

All settings for the headless CLI pipeline live in [`config.yaml`](config.yaml):

| Section | What it controls |
|---|---|
| `data` | Input file path, target column, index column, columns to drop |
| `preprocessing` | Imputation strategy, scaling method, log transform, test split size |
| `eda` | Correlation threshold, PCA components, top-N features/categories |
| `models` | List of models to train |
| `model_training` | Task type (auto/classification/regression), CV folds, scoring metric, tuning toggle, custom hyperparameter grids |
| `output` | Results directory, save plots/models toggles |

---

## Supported Models

| Classification | Regression |
|---|---|
| Random Forest | Random Forest |
| Logistic Regression | Ridge Regression |
| SVM (SVC) | SVR |
| K-Nearest Neighbors | K-Nearest Neighbors |

---

## Requirements

- Python 3.9+
- pandas, scikit-learn, seaborn, matplotlib, pyyaml, joblib, shiny, shinyswatch

---

## Data

The included sample dataset is the [Wisconsin Breast Cancer Diagnostic Dataset (WBCD)](https://doi.org/10.24432/C5DW2B), donated by Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1995) to the UCI Machine Learning Repository. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## License

[MIT](LICENSE)
