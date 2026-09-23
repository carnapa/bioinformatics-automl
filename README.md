# Bioinformatics AutoML Pipeline

A config-driven machine learning pipeline for bioinformatics datasets. Point it at a CSV, edit one YAML file, and get end-to-end results: exploratory data analysis, preprocessing, model training with hyperparameter tuning, and evaluation reports with visualisations.

## Features

- **Declarative configuration** — control every stage from a single `config.yaml`
- **Automated EDA** — statistical summaries, correlation heatmaps, class-balance plots, PCA with configurable components
- **Preprocessing** — missing-value imputation (mean/median/zero), log transform, scaling (standard/minmax/robust)
- **Auto task detection** — automatically identifies classification vs. regression based on target column
- **Model training** — hyperparameter tuning via GridSearchCV with stratified cross-validation
- **Evaluation** — classification reports, confusion matrices, ROC curves, feature importance plots, and a comparison CSV
- **Model persistence** — trained models saved as `.joblib` files for later use

## Project Structure

```
bioinformatics-automl/
├── main.py              # Pipeline entry point
├── config.yaml          # All pipeline settings
├── requirements.txt     # Python dependencies
├── data/                # Input datasets
├── results/             # Generated outputs (plots, reports, models)
├── src/
│   ├── utils.py         # Config & data loading utilities
│   ├── eda.py           # Exploratory data analysis module
│   └── models.py        # Model training, tuning & evaluation
└── notebooks/           # Jupyter notebooks (exploratory work in progress)
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/carnapa/bioinformatics-automl.git
cd bioinformatics-automl

# Install dependencies
pip install -r requirements.txt

# Edit config.yaml to point to your dataset, then run:
python main.py
```

Results (plots, reports, saved models) are written to the `results/` directory.

## Configuration

All settings live in [`config.yaml`](config.yaml):

| Section | What it controls |
|---|---|
| `data` | Input file path, target column, index column, columns to drop |
| `preprocessing` | Imputation strategy, scaling method, log transform, test split size |
| `eda` | Correlation threshold, PCA components, top-N features/categories |
| `models` | List of models to train |
| `model_training` | Task type (auto/classification/regression), CV folds, scoring metric, tuning toggle, custom hyperparameter grids |
| `output` | Results directory, save plots/models toggles |

## Supported Models

| Classification | Regression |
|---|---|
| Random Forest | Random Forest |
| Logistic Regression | Ridge Regression |
| SVM (SVC) | SVR |
| K-Nearest Neighbors | K-Nearest Neighbors |

## Sample Output

After running the pipeline you get:

- `model_comparison.csv` — side-by-side metrics for all models
- `<model>_report.txt` — detailed classification/regression report
- `<model>_confusion_matrix.png` — confusion matrix heatmap
- `<model>_roc_curve.png` — ROC curve with AUC (binary classification)
- `<model>_feature_importance.png` — top feature importances
- `<model>_model.joblib` — serialised trained model
- EDA outputs: `data_summary.csv`, `correlation_heatmap.png`, `pca_*.png`, `target_distribution.png`

## Requirements

- Python 3.9+
- pandas, scikit-learn, seaborn, matplotlib, pyyaml, joblib

## License

[MIT](LICENSE)
