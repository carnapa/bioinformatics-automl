import io
import asyncio
import zipfile
from pathlib import Path
import pandas as pd
import joblib
import shinyswatch
from shiny import reactive
from shiny.express import input, render, ui

from src.eda import (
    generate_summary,
    generate_target_distribution_plot,
    generate_correlation_heatmap,
    generate_pca_plots,
)
from src.models import run_model_training

# Set page title and theme
ui.page_opts(
    title="Bioinformatics AutoML Dashboard",
    fillable=True,
    theme=shinyswatch.theme.flatly
)

# ---------------------------------------------------------------------------
# Reactive State
# ---------------------------------------------------------------------------
loaded_df = reactive.value(None)

# ---------------------------------------------------------------------------
# Section 5: Non-Blocking Model Training (Extended Task)
# ---------------------------------------------------------------------------
@reactive.extended_task
async def train_models_task(df: pd.DataFrame, config: dict):
    """Offloads CPU-heavy GridSearchCV model training to a background worker thread."""
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, run_model_training, df, config)
    return results


@reactive.calc
def training_results():
    """Reactive calc returning results when extended_task finishes successfully."""
    if train_models_task.status() == "success":
        return train_models_task.result()
    return None


@reactive.effect
def _handle_task_notifications():
    """Display toast notifications on task state changes."""
    status = train_models_task.status()
    if status == "running":
        ui.notification_show("Model training in progress with cross-validation...", duration=6, type="message")
    elif status == "success":
        ui.notification_show("Model training completed successfully!", duration=5, type="message")
    elif status == "error":
        ui.notification_show(f"Training failed: {train_models_task.error()}", duration=None, type="error")


# ---------------------------------------------------------------------------
# Sidebar Layout
# ---------------------------------------------------------------------------
with ui.sidebar(width=340, bg="#f8f9fa", title="Pipeline Controls"):
    ui.h5("1. Data Input")
    ui.input_file("file_upload", "Upload Dataset (.csv / .xlsx)", accept=[".csv", ".xlsx"], multiple=False)
    ui.input_action_button("load_sample_btn", "Load Sample Dataset (Breast Cancer)", class_="btn-outline-primary btn-sm mb-3")

    ui.hr()
    ui.h5("2. Target & Columns")
    ui.input_select("target_col", "Target Variable", choices=[])
    ui.input_selectize("drop_cols", "Columns to Exclude", choices=[], multiple=True)

    ui.hr()
    ui.h5("3. Preprocessing")
    ui.input_select("task_type", "Task Type", choices=["auto", "classification", "regression"])
    ui.input_select("scaling", "Feature Scaling", choices=["standard", "minmax", "robust"])
    ui.input_select("impute_missing", "Missing Values Imputation", choices=["median", "mean", "zero"])
    ui.input_switch("log_transform", "Log Transform (Positive Features)", value=True)
    ui.input_slider("test_size", "Test Split Size", min=0.1, max=0.5, value=0.2, step=0.05)

    ui.hr()
    ui.h5("4. Model Training")
    ui.input_checkbox_group(
        "selected_models",
        "Select Algorithms",
        choices={
            "random_forest": "Random Forest",
            "logistic_regression": "Logistic Regression / Ridge",
            "svm": "Support Vector Machine (SVM / SVR)",
            "knn": "K-Nearest Neighbors (KNN)",
        },
        selected=["random_forest", "logistic_regression", "svm", "knn"]
    )
    ui.input_numeric("cv_folds", "Cross-Validation Folds", value=5, min=2, max=10)
    ui.input_switch("use_tuning", "Hyperparameter Tuning (GridSearch)", value=True)

    ui.input_task_button("train_btn", "Train Models", class_="btn-success w-100 mt-2")


# ---------------------------------------------------------------------------
# Data Loading & Event Handlers
# ---------------------------------------------------------------------------
@reactive.effect
@reactive.event(input.load_sample_btn)
def _load_sample_dataset():
    path = Path("data/data_breast_cancer.csv")
    if path.exists():
        df = pd.read_csv(path)
        if "id" in df.columns:
            df = df.set_index("id")
        loaded_df.set(df)
        ui.notification_show("Loaded sample Breast Cancer dataset!", type="message", duration=3)
    else:
        ui.notification_show("Sample file 'data/data_breast_cancer.csv' not found.", type="error")


@reactive.effect
@reactive.event(input.file_upload)
def _handle_file_upload():
    file_info = input.file_upload()
    if not file_info:
        return
    file_path = Path(file_info[0]["datapath"])
    name = file_info[0]["name"]
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        loaded_df.set(df)
        ui.notification_show(f"Uploaded '{name}' successfully!", type="message", duration=3)
    except Exception as e:
        ui.notification_show(f"Error reading file: {e}", type="error")


@reactive.effect
def _sync_column_selectors():
    df = loaded_df()
    if df is None:
        return
    cols = list(df.columns)
    selected_target = "diagnosis" if "diagnosis" in cols else (cols[0] if cols else None)
    ui.update_select("target_col", choices=cols, selected=selected_target)

    default_drops = [c for c in ["Unnamed: 32"] if c in cols]
    other_cols = [c for c in cols if c != selected_target]
    ui.update_selectize("drop_cols", choices=other_cols, selected=default_drops)


@reactive.calc
def cleaned_df():
    df = loaded_df()
    if df is None:
        return None
    drops = input.drop_cols() or []
    existing_drops = [c for c in drops if c in df.columns]
    if existing_drops:
        return df.drop(columns=existing_drops)
    return df.copy()


# ---------------------------------------------------------------------------
# Training Trigger (Invokes Extended Task)
# ---------------------------------------------------------------------------
@reactive.effect
@reactive.event(input.train_btn)
def _trigger_training():
    df = cleaned_df()
    if df is None:
        ui.notification_show("Please load or upload a dataset first.", type="warning")
        return

    target = input.target_col()
    if not target or target not in df.columns:
        ui.notification_show("Please select a valid target column.", type="warning")
        return

    models = input.selected_models()
    if not models:
        ui.notification_show("Please select at least one model to train.", type="warning")
        return

    custom_config = {
        "data": {
            "target_column": target,
            "drop_columns": input.drop_cols() or [],
        },
        "preprocessing": {
            "log_transform": input.log_transform(),
            "scaling": input.scaling(),
            "impute_missing": input.impute_missing(),
            "test_size": float(input.test_size()),
        },
        "models": list(models),
        "model_training": {
            "task_type": input.task_type(),
            "cv_folds": int(input.cv_folds()),
            "use_tuning": input.use_tuning(),
            "scoring_metric": "f1_weighted" if input.task_type() != "regression" else "r2",
        },
        "output": {
            "dir": "results/",
            "save_plots": False,
            "save_models": False,
        }
    }

    # Launch background async task
    train_models_task(df, custom_config)


# ---------------------------------------------------------------------------
# Section 6: Helper for Zip Results Bundle & Model Downloads
# ---------------------------------------------------------------------------
def _create_results_zip(results, df=None) -> bytes:
    """Creates an in-memory zip file containing all reports, plots, models, and CSVs."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if results:
            # 1. Comparison CSV
            summary_rows = [
                {k: v for k, v in r.items() if k not in ("figures", "report", "model_instance")}
                for r in results
            ]
            zf.writestr("model_comparison.csv", pd.DataFrame(summary_rows).to_csv(index=False))

            # 2. Per-model artifacts
            for r in results:
                m_name = r["model"]
                if "report" in r and r["report"]:
                    zf.writestr(f"{m_name}_report.txt", r["report"])
                for fig_name, fig in r.get("figures", {}).items():
                    if fig is not None:
                        img_buf = io.BytesIO()
                        fig.savefig(img_buf, format="png", dpi=150)
                        img_buf.seek(0)
                        zf.writestr(f"{m_name}_{fig_name}.png", img_buf.read())
                if "model_instance" in r and r["model_instance"] is not None:
                    joblib_buf = io.BytesIO()
                    joblib.dump(r["model_instance"], joblib_buf)
                    joblib_buf.seek(0)
                    zf.writestr(f"{m_name}_model.joblib", joblib_buf.read())

        if df is not None:
            zf.writestr("data_summary.csv", generate_summary(df).to_csv())

    buf.seek(0)
    return buf.getvalue()


def _get_model_joblib_bytes(model_name: str) -> bytes:
    results = training_results() or []
    for r in results:
        if r["model"] == model_name and r.get("model_instance") is not None:
            buf = io.BytesIO()
            joblib.dump(r["model_instance"], buf)
            buf.seek(0)
            return buf.read()
    return b""


# ---------------------------------------------------------------------------
# Main Tabs Layout
# ---------------------------------------------------------------------------
with ui.navset_card_tab(id="main_tabs"):

    # TAB 1: DATA PREVIEW & SUMMARY
    with ui.nav_panel("Data Preview & Stats"):
        with ui.layout_columns(col_widths=[12]):
            with ui.card():
                ui.card_header("Dataset Preview (First 50 Rows)")
                @render.data_frame
                def preview_table():
                    df = cleaned_df()
                    if df is None:
                        return None
                    return render.DataGrid(df.head(50), height="350px", width="100%")

            with ui.card():
                ui.card_header("Statistical Summary")
                @render.data_frame
                def summary_table():
                    df = cleaned_df()
                    if df is None:
                        return None
                    summary_df = generate_summary(df).reset_index().rename(columns={"index": "feature"})
                    return render.DataGrid(summary_df, height="350px", width="100%")

    # TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
    with ui.nav_panel("Exploratory Data Analysis"):
        with ui.layout_columns(col_widths=[6, 6]):
            with ui.card():
                ui.card_header("Target Variable Distribution")
                @render.plot
                def plot_target_dist():
                    df = cleaned_df()
                    target = input.target_col()
                    if df is None or not target:
                        return None
                    return generate_target_distribution_plot(df, target)

            with ui.card():
                ui.card_header("Correlation Heatmap")
                @render.plot
                def plot_correlation():
                    df = cleaned_df()
                    if df is None:
                        return None
                    return generate_correlation_heatmap(df)

        with ui.card():
            ui.card_header("Principal Component Analysis (PCA: PC1 vs PC2)")
            @render.plot
            def plot_pca():
                df = cleaned_df()
                target = input.target_col()
                if df is None or not target:
                    return None
                plots, _ = generate_pca_plots(df, target=target, n_components=2, impute_strategy=input.impute_missing())
                return plots.get("pca_PC1_vs_PC2")

    # TAB 3: MODEL TRAINING & EVALUATION
    with ui.nav_panel("Models & Leaderboard"):
        with ui.card():
            ui.card_header("Model Comparison Leaderboard")
            @render.data_frame
            def comparison_table():
                results = training_results()
                if not results:
                    return None
                summary_rows = [
                    {k: v for k, v in r.items() if k not in ("figures", "report", "model_instance")}
                    for r in results
                ]
                return render.DataGrid(pd.DataFrame(summary_rows), width="100%")

            with ui.layout_columns(col_widths=[6, 6]):
                @render.download_button(
                    filename="model_comparison.csv",
                    label="Download Comparison CSV"
                )
                async def download_comparison():
                    results = training_results()
                    if results:
                        summary_rows = [
                            {k: v for k, v in r.items() if k not in ("figures", "report", "model_instance")}
                            for r in results
                        ]
                        df = pd.DataFrame(summary_rows)
                        yield df.to_csv(index=False)

                @render.download_button(
                    filename="bioinformatics_automl_results.zip",
                    label="Download Full Results Bundle (.zip)"
                )
                async def download_bundle():
                    results = training_results()
                    df = cleaned_df()
                    if results:
                        yield _create_results_zip(results, df)

        ui.h4("Individual Model Inspection", class_="mt-3 mb-2")

        with ui.accordion(id="acc_models", open=False):
            with ui.accordion_panel("Random Forest"):
                @render.text
                def rf_params():
                    return _get_best_params("random_forest")
                @render.text
                def rf_report():
                    return _get_report("random_forest")
                @render.download_button(
                    filename="random_forest_model.joblib",
                    label="Download Random Forest Model (.joblib)"
                )
                async def download_rf_model():
                    yield _get_model_joblib_bytes("random_forest")
                with ui.layout_columns(col_widths=[4, 4, 4]):
                    @render.plot
                    def rf_cm():
                        return _get_figure("random_forest", "confusion_matrix")
                    @render.plot
                    def rf_roc():
                        return _get_figure("random_forest", "roc_curve")
                    @render.plot
                    def rf_fi():
                        return _get_figure("random_forest", "feature_importance")

            with ui.accordion_panel("Logistic Regression / Ridge"):
                @render.text
                def lr_params():
                    return _get_best_params("logistic_regression")
                @render.text
                def lr_report():
                    return _get_report("logistic_regression")
                @render.download_button(
                    filename="logistic_regression_model.joblib",
                    label="Download Logistic Regression Model (.joblib)"
                )
                async def download_lr_model():
                    yield _get_model_joblib_bytes("logistic_regression")
                with ui.layout_columns(col_widths=[4, 4, 4]):
                    @render.plot
                    def lr_cm():
                        return _get_figure("logistic_regression", "confusion_matrix")
                    @render.plot
                    def lr_roc():
                        return _get_figure("logistic_regression", "roc_curve")
                    @render.plot
                    def lr_fi():
                        return _get_figure("logistic_regression", "feature_importance")

            with ui.accordion_panel("Support Vector Machine (SVM / SVR)"):
                @render.text
                def svm_params():
                    return _get_best_params("svm")
                @render.text
                def svm_report():
                    return _get_report("svm")
                @render.download_button(
                    filename="svm_model.joblib",
                    label="Download SVM Model (.joblib)"
                )
                async def download_svm_model():
                    yield _get_model_joblib_bytes("svm")
                with ui.layout_columns(col_widths=[4, 4, 4]):
                    @render.plot
                    def svm_cm():
                        return _get_figure("svm", "confusion_matrix")
                    @render.plot
                    def svm_roc():
                        return _get_figure("svm", "roc_curve")
                    @render.plot
                    def svm_fi():
                        return _get_figure("svm", "feature_importance")

            with ui.accordion_panel("K-Nearest Neighbors (KNN)"):
                @render.text
                def knn_params():
                    return _get_best_params("knn")
                @render.text
                def knn_report():
                    return _get_report("knn")
                @render.download_button(
                    filename="knn_model.joblib",
                    label="Download KNN Model (.joblib)"
                )
                async def download_knn_model():
                    yield _get_model_joblib_bytes("knn")
                with ui.layout_columns(col_widths=[4, 4, 4]):
                    @render.plot
                    def knn_cm():
                        return _get_figure("knn", "confusion_matrix")
                    @render.plot
                    def knn_roc():
                        return _get_figure("knn", "roc_curve")
                    @render.plot
                    def knn_fi():
                        return _get_figure("knn", "feature_importance")


# ---------------------------------------------------------------------------
# Helper functions to extract outputs from reactive training_results
# ---------------------------------------------------------------------------
def _get_best_params(model_name: str) -> str:
    results = training_results() or []
    for r in results:
        if r["model"] == model_name:
            return f"Best Hyperparameters: {r.get('best_params', 'N/A')} (CV Score: {r.get('best_cv_score', 'N/A')})"
    return "Model not trained yet."

def _get_report(model_name: str) -> str:
    results = training_results() or []
    for r in results:
        if r["model"] == model_name:
            return r.get("report", "")
    return ""

def _get_figure(model_name: str, fig_key: str):
    results = training_results() or []
    for r in results:
        if r["model"] == model_name:
            return r.get("figures", {}).get(fig_key)
    return None
