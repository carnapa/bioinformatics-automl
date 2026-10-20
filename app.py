import io
import asyncio
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import torch
import shinyswatch
from sklearn.decomposition import PCA
from shiny import reactive
from shiny.express import input, render, ui

from src.eda import (
    generate_summary,
    generate_target_distribution_plot,
    generate_correlation_heatmap,
    generate_pca_plots,
)
from src.automl_models import run_model_training
from src.bio.rnaseq import RNASeqPipeline
from src.models.dl_expression import (
    ExpressionMLP, ExpressionTrainer,
    ExpressionVAE, VAETrainer,
)

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

# Reactive state for the Transcriptomics tab — shared between both new tabs
rnaseq_counts_df   = reactive.value(None)   # raw count matrix (DataFrame)
rnaseq_metadata_df = reactive.value(None)   # sample metadata (DataFrame)

# ---------------------------------------------------------------------------
# Extended Task 1: AutoML sklearn model training (unchanged)
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
    """Display toast notifications on AutoML task state changes."""
    status = train_models_task.status()
    if status == "running":
        ui.notification_show("Model training in progress with cross-validation...", duration=6, type="message")
    elif status == "success":
        ui.notification_show("Model training completed successfully!", duration=5, type="message")
    elif status == "error":
        ui.notification_show(f"Training failed: {train_models_task.error()}", duration=None, type="error")


# ---------------------------------------------------------------------------
# Extended Task 2: RNA-Seq analysis (normalise + DEA + plots)
# ---------------------------------------------------------------------------

def _run_rnaseq(counts_df, metadata_df, target_col, group_a, group_b,
                log2fc_thresh, fdr_thresh):
    """
    Synchronous RNA-Seq worker — runs in a thread pool executor.

    Returns a dict with keys:
        dea_df       : pd.DataFrame  — full DEA results table
        volcano_fig  : Figure        — volcano plot
        ma_fig       : Figure        — MA plot
        log_cpm      : pd.DataFrame  — normalised expression matrix
        metadata     : pd.DataFrame  — aligned sample metadata
    """
    pipe = RNASeqPipeline(counts_df, metadata_df, target_col=target_col)
    pipe.normalize_counts()

    dea_df = pipe.run_dea(
        group_a=group_a,
        group_b=group_b,
        fdr_thresh=fdr_thresh,
        log2fc_thresh=log2fc_thresh,
    )

    volcano_fig = pipe.plot_volcano(dea_df, log2fc_thresh=log2fc_thresh, fdr_thresh=fdr_thresh)
    ma_fig      = pipe.plot_ma(dea_df, fdr_thresh=fdr_thresh)

    return {
        "dea_df":      dea_df,
        "volcano_fig": volcano_fig,
        "ma_fig":      ma_fig,
        "log_cpm":     pipe.log_cpm,
        "metadata":    pipe.metadata,
    }


@reactive.extended_task
async def rnaseq_task(counts_df, metadata_df, target_col, group_a, group_b,
                      log2fc_thresh, fdr_thresh):
    """Offloads RNA-Seq normalisation + DEA to a background worker thread."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, _run_rnaseq,
        counts_df, metadata_df, target_col, group_a, group_b,
        log2fc_thresh, fdr_thresh,
    )
    return result


@reactive.calc
def rnaseq_results():
    """Return RNA-Seq task result dict when the task completes successfully."""
    if rnaseq_task.status() == "success":
        return rnaseq_task.result()
    return None


@reactive.effect
def _handle_rnaseq_notifications():
    """Display toast notifications on RNA-Seq task state changes."""
    status = rnaseq_task.status()
    if status == "running":
        ui.notification_show("RNA-Seq analysis in progress...", duration=6, type="message")
    elif status == "success":
        res = rnaseq_task.result()
        n_sig = int(res["dea_df"]["significant"].sum())
        ui.notification_show(
            f"RNA-Seq analysis complete! {n_sig} significant genes found.",
            duration=5, type="message"
        )
    elif status == "error":
        ui.notification_show(
            f"RNA-Seq analysis failed: {rnaseq_task.error()}",
            duration=None, type="error"
        )


# ---------------------------------------------------------------------------
# Extended Task 3: PyTorch DL training (MLP or VAE)
# ---------------------------------------------------------------------------

def _run_dl(log_cpm, metadata, target_col, group_b,
            model_type, hidden_dims, latent_dim,
            epochs, lr, dropout, batch_size, patience):
    """
    Synchronous PyTorch training worker — runs in a thread pool executor.

    Returns a dict with keys (varies by model_type):
        model_type   : str
        curves_fig   : Figure    — training loss curves
        model_bytes  : bytes     — serialised .pt state dict
        eval_metrics : dict      — {loss, accuracy} or {loss, r2}  [MLP only]
        latent_fig   : Figure    — VAE PCA scatter plot             [VAE only]
    """
    X_arr    = log_cpm.values.astype("float32")
    input_dim = X_arr.shape[1]

    result = {"model_type": model_type}

    if model_type == "mlp":
        # Binary label: group_b = 1, everything else = 0
        y_arr = (metadata[target_col] == group_b).astype(int).loc[log_cpm.index].values

        split = max(1, int(len(X_arr) * 0.8))
        X_train, X_val = X_arr[:split], X_arr[split:]
        y_train, y_val = y_arr[:split], y_arr[split:]

        model   = ExpressionMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=2,
            dropout_rate=dropout,
        )
        trainer = ExpressionTrainer(model, task="classification")
        history = trainer.fit(
            X_train, y_train, X_val, y_val,
            epochs=epochs, lr=lr, batch_size=batch_size, patience=patience,
        )

        result["curves_fig"]   = trainer.plot_training_curves(history)
        result["eval_metrics"] = trainer.evaluate(X_val, y_val)

        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        result["model_bytes"] = buf.getvalue()

    elif model_type == "vae":
        vae     = ExpressionVAE(input_dim=input_dim, hidden_dims=hidden_dims,
                                latent_dim=latent_dim)
        trainer = VAETrainer(vae)
        history = trainer.fit_vae(
            X_arr, epochs=epochs, lr=lr, batch_size=batch_size,
        )

        result["curves_fig"] = trainer.plot_training_curves(history)

        # PCA projection of latent mu vectors, coloured by condition
        mu_vecs  = trainer.transform(X_arr)
        pca      = PCA(n_components=2)
        coords   = pca.fit_transform(mu_vecs)
        conditions = metadata[target_col].values
        unique_conds = np.unique(conditions)
        colours  = plt.cm.tab10(np.linspace(0, 0.5, len(unique_conds)))

        fig, ax = plt.subplots(figsize=(7, 5))
        for cond, col in zip(unique_conds, colours):
            mask = conditions == cond
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       label=cond, c=[col], alpha=0.8, s=60,
                       edgecolors='k', linewidths=0.4)
        ev = pca.explained_variance_ratio_ * 100
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
        ax.set_title("VAE Latent Space — PCA Projection")
        ax.legend(title=target_col, framealpha=0.8)
        fig.tight_layout()
        result["latent_fig"] = fig

        buf = io.BytesIO()
        torch.save(vae.state_dict(), buf)
        result["model_bytes"] = buf.getvalue()

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'mlp' or 'vae'.")

    return result


@reactive.extended_task
async def dl_task(log_cpm, metadata, target_col, group_b,
                  model_type, hidden_dims, latent_dim,
                  epochs, lr, dropout, batch_size, patience):
    """Offloads PyTorch model training to a background worker thread."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, _run_dl,
        log_cpm, metadata, target_col, group_b,
        model_type, hidden_dims, latent_dim,
        epochs, lr, dropout, batch_size, patience,
    )
    return result


@reactive.calc
def dl_results():
    """Return DL task result dict when training completes successfully."""
    if dl_task.status() == "success":
        return dl_task.result()
    return None


@reactive.effect
def _handle_dl_notifications():
    """Display toast notifications on DL task state changes."""
    status = dl_task.status()
    if status == "running":
        ui.notification_show("PyTorch model training in progress...", duration=6, type="message")
    elif status == "success":
        ui.notification_show("PyTorch training complete!", duration=5, type="message")
    elif status == "error":
        ui.notification_show(
            f"PyTorch training failed: {dl_task.error()}",
            duration=None, type="error"
        )


# ---------------------------------------------------------------------------
# Sidebar Layout (unchanged)
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
# Data Loading & Event Handlers (unchanged)
# ---------------------------------------------------------------------------
@reactive.effect
@reactive.event(input.load_sample_btn)
def _load_sample_dataset():
    path = Path(__file__).parent / "data" / "data_breast_cancer.csv"
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
# Training Trigger — AutoML (unchanged)
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
# Helper for Zip Results Bundle & Model Downloads (unchanged)
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
# Transcriptomics tab — reactive event handlers
# ---------------------------------------------------------------------------

@reactive.effect
@reactive.event(input.rna_counts_upload)
def _handle_counts_upload():
    """Load uploaded count matrix CSV into reactive state."""
    file_info = input.rna_counts_upload()
    if not file_info:
        return
    try:
        df = pd.read_csv(Path(file_info[0]["datapath"]), index_col=0)
        rnaseq_counts_df.set(df)
        ui.notification_show(
            f"Count matrix loaded: {df.shape[0]} rows × {df.shape[1]} columns.",
            type="message", duration=3
        )
    except Exception as e:
        ui.notification_show(f"Error reading count matrix: {e}", type="error")


@reactive.effect
@reactive.event(input.rna_metadata_upload)
def _handle_metadata_upload():
    """Load uploaded metadata CSV and populate the condition column dropdown."""
    file_info = input.rna_metadata_upload()
    if not file_info:
        return
    try:
        df = pd.read_csv(Path(file_info[0]["datapath"]), index_col=0)
        rnaseq_metadata_df.set(df)
        cols = list(df.columns)
        ui.update_select("rna_condition_col", choices=cols,
                         selected=cols[0] if cols else None)
        ui.notification_show(
            f"Metadata loaded: {df.shape[0]} samples, columns: {cols}.",
            type="message", duration=3
        )
    except Exception as e:
        ui.notification_show(f"Error reading metadata: {e}", type="error")


@reactive.effect
def _sync_group_selectors():
    """Populate Group A / Group B dropdowns from the selected condition column."""
    meta = rnaseq_metadata_df()
    try:
        col = input.rna_condition_col()
    except Exception:
        col = None
    if meta is None or not col or col not in meta.columns:
        return
    unique_vals = sorted(meta[col].dropna().unique().tolist())
    sel_a = unique_vals[0] if len(unique_vals) > 0 else None
    sel_b = unique_vals[1] if len(unique_vals) > 1 else sel_a
    ui.update_select("rna_group_a", choices=unique_vals, selected=sel_a)
    ui.update_select("rna_group_b", choices=unique_vals, selected=sel_b)


@reactive.effect
@reactive.event(input.rna_run_btn)
def _trigger_rnaseq():
    """Validate inputs and launch the RNA-Seq background task."""
    counts = rnaseq_counts_df()
    meta   = rnaseq_metadata_df()
    if counts is None:
        ui.notification_show("Please upload a count matrix CSV first.", type="warning")
        return
    if meta is None:
        ui.notification_show("Please upload a metadata CSV first.", type="warning")
        return

    target_col = input.rna_condition_col()
    group_a    = input.rna_group_a()
    group_b    = input.rna_group_b()

    if not target_col or not group_a or not group_b:
        ui.notification_show("Please select condition column, Group A, and Group B.", type="warning")
        return
    if group_a == group_b:
        ui.notification_show("Group A and Group B must be different.", type="warning")
        return

    rnaseq_task(
        counts, meta, target_col, group_a, group_b,
        float(input.rna_log2fc_thresh()),
        float(input.rna_fdr_thresh()),
    )


# ---------------------------------------------------------------------------
# PyTorch DL tab — reactive event handler
# ---------------------------------------------------------------------------

@reactive.effect
@reactive.event(input.dl_train_btn)
def _trigger_dl_training():
    """Validate RNA-Seq results and launch the DL background task."""
    rna_res = rnaseq_results()
    if rna_res is None:
        ui.notification_show(
            "Run RNA-Seq analysis first (Transcriptomics & DEA tab).", type="warning"
        )
        return

    log_cpm  = rna_res["log_cpm"]
    metadata = rna_res["metadata"]

    # Parse hidden_dims from the comma-separated text input
    try:
        hidden_dims = [int(x.strip()) for x in input.dl_hidden_dims().split(",") if x.strip()]
        if not hidden_dims:
            raise ValueError("Empty list")
    except Exception:
        ui.notification_show(
            "Hidden Dims must be comma-separated integers, e.g. '512,256,128'.", type="warning"
        )
        return

    model_type = input.dl_model_type()

    # Determine target_col and group_b from the RNA-Seq inputs
    target_col = input.rna_condition_col()
    group_b    = input.rna_group_b()

    dl_task(
        log_cpm, metadata, target_col, group_b,
        model_type, hidden_dims,
        int(input.dl_latent_dim()),
        int(input.dl_epochs()),
        float(input.dl_lr()),
        float(input.dl_dropout()),
        int(input.dl_batch_size()),
        int(input.dl_patience()),
    )


# ---------------------------------------------------------------------------
# Main Tabs Layout
# ---------------------------------------------------------------------------
with ui.navset_card_tab(id="main_tabs"):

    # ------------------------------------------------------------------
    # TAB 1: DATA PREVIEW & SUMMARY (unchanged)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # TAB 2: EXPLORATORY DATA ANALYSIS (unchanged)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # TAB 3: MODEL TRAINING & EVALUATION (unchanged)
    # ------------------------------------------------------------------
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
                @render.download(filename="model_comparison.csv")
                async def download_comparison():
                    results = training_results()
                    if results:
                        summary_rows = [
                            {k: v for k, v in r.items() if k not in ("figures", "report", "model_instance")}
                            for r in results
                        ]
                        df = pd.DataFrame(summary_rows)
                        yield df.to_csv(index=False)

                @render.download(filename="bioinformatics_automl_results.zip")
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
                @render.download(filename="random_forest_model.joblib")
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
                @render.download(filename="logistic_regression_model.joblib")
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
                @render.download(filename="svm_model.joblib")
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
                @render.download(filename="knn_model.joblib")
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

    # ------------------------------------------------------------------
    # TAB 4: TRANSCRIPTOMICS & DEA  (new)
    # ------------------------------------------------------------------
    with ui.nav_panel("Transcriptomics & DEA"):

        # Control panel — inputs at the top of the tab
        with ui.card(class_="mb-3"):
            ui.card_header("RNA-Seq Analysis Controls")
            with ui.layout_columns(col_widths=[3, 3, 3, 3]):
                ui.input_file(
                    "rna_counts_upload",
                    "Count Matrix CSV",
                    accept=[".csv"],
                    multiple=False,
                )
                ui.input_file(
                    "rna_metadata_upload",
                    "Sample Metadata CSV",
                    accept=[".csv"],
                    multiple=False,
                )
                ui.input_select(
                    "rna_condition_col",
                    "Condition Column",
                    choices=[],
                )
                ui.div()   # spacer

            with ui.layout_columns(col_widths=[3, 3, 3, 3]):
                ui.input_select("rna_group_a", "Group A (Reference)", choices=[])
                ui.input_select("rna_group_b", "Group B (Comparison)", choices=[])
                ui.input_slider(
                    "rna_log2fc_thresh", "Log2FC Threshold",
                    min=0.0, max=3.0, value=1.0, step=0.1,
                )
                ui.input_slider(
                    "rna_fdr_thresh", "FDR Threshold",
                    min=0.01, max=0.20, value=0.05, step=0.01,
                )

            ui.input_task_button(
                "rna_run_btn",
                "Run RNA-Seq Analysis",
                class_="btn-primary mt-2",
            )

        # Output panels
        with ui.layout_columns(col_widths=[6, 6]):
            with ui.card():
                ui.card_header("Volcano Plot")
                @render.plot
                def plot_volcano():
                    res = rnaseq_results()
                    if res is None:
                        return None
                    return res["volcano_fig"]

            with ui.card():
                ui.card_header("MA Plot")
                @render.plot
                def plot_ma():
                    res = rnaseq_results()
                    if res is None:
                        return None
                    return res["ma_fig"]

        with ui.card():
            ui.card_header("Top Differentially Expressed Genes (Top 50 by Adjusted p-value)")

            @render.data_frame
            def dea_table():
                res = rnaseq_results()
                if res is None:
                    return None
                top50 = (
                    res["dea_df"]
                    .sort_values("adj_p_value")
                    .head(50)
                    .reset_index(drop=True)
                )
                return render.DataGrid(top50, height="350px", width="100%")

            @render.download(filename="dea_results.csv")
            async def download_dea():
                """Download the full DEA results table as CSV."""
                res = rnaseq_results()
                if res is not None:
                    yield res["dea_df"].to_csv(index=False)

    # ------------------------------------------------------------------
    # TAB 5: PYTORCH EXPRESSION DL  (new)
    # ------------------------------------------------------------------
    with ui.nav_panel("PyTorch Expression DL"):

        # Status message when RNA-Seq has not been run yet
        @render.ui
        def dl_prereq_notice():
            if rnaseq_results() is None:
                return ui.div(
                    ui.tags.div(
                        ui.tags.strong("RNA-Seq analysis required. "),
                        "Please run the ",
                        ui.tags.em("Transcriptomics & DEA"),
                        " analysis first — this tab uses the normalised "
                        "expression matrix produced there.",
                        class_="alert alert-info",
                    )
                )
            rna = rnaseq_results()
            n_genes   = rna["log_cpm"].shape[1]
            n_samples = rna["log_cpm"].shape[0]
            return ui.div(
                ui.tags.div(
                    ui.tags.strong("Ready. "),
                    f"Normalised expression matrix: {n_samples} samples × {n_genes} genes.",
                    class_="alert alert-success",
                )
            )

        # Control panel
        with ui.card(class_="mb-3"):
            ui.card_header("PyTorch Model Controls")

            with ui.layout_columns(col_widths=[4, 4, 4]):
                ui.input_select(
                    "dl_model_type",
                    "Model Type",
                    choices={
                        "mlp": "MLP Classifier",
                        "vae": "VAE Dimensionality Reduction",
                    },
                    selected="mlp",
                )
                ui.input_text(
                    "dl_hidden_dims",
                    "Hidden Layer Dims (comma-separated)",
                    value="512,256,128",
                )
                ui.input_numeric(
                    "dl_latent_dim",
                    "Latent Dimensions (VAE only)",
                    value=32, min=2, max=256,
                )

            with ui.layout_columns(col_widths=[3, 3, 3, 3]):
                ui.input_numeric("dl_epochs",     "Epochs",              value=100, min=1,   max=1000)
                ui.input_numeric("dl_lr",         "Learning Rate",       value=0.001, min=1e-5, max=1.0, step=0.0001)
                ui.input_numeric("dl_dropout",    "Dropout Rate",        value=0.3,   min=0.0,  max=0.9, step=0.05)
                ui.input_numeric("dl_batch_size", "Batch Size",          value=32,    min=4,    max=512)

            with ui.layout_columns(col_widths=[4, 8]):
                ui.input_numeric("dl_patience", "Early Stopping Patience (MLP)", value=10, min=1, max=100)
                ui.div()   # spacer

            ui.input_task_button(
                "dl_train_btn",
                "Train PyTorch Model",
                class_="btn-success mt-2",
            )

        # Output panels
        with ui.layout_columns(col_widths=[6, 6]):
            with ui.card():
                ui.card_header("Training Loss Curves")
                @render.plot
                def dl_loss_curves():
                    res = dl_results()
                    if res is None:
                        return None
                    return res.get("curves_fig")

            with ui.card():
                ui.card_header("VAE Latent Space Projection (PCA)")
                @render.plot
                def dl_latent_plot():
                    res = dl_results()
                    if res is None or res.get("model_type") != "vae":
                        return None
                    return res.get("latent_fig")

        with ui.card():
            ui.card_header("Model Evaluation")

            @render.text
            def dl_metrics_text():
                res = dl_results()
                if res is None:
                    return "Train a model to see evaluation metrics."
                mtype = res.get("model_type", "")
                if mtype == "mlp":
                    m = res.get("eval_metrics", {})
                    loss = m.get("loss", float("nan"))
                    acc  = m.get("accuracy", float("nan"))
                    return f"Model: MLP Classifier\nValidation Loss: {loss:.4f}\nValidation Accuracy: {acc:.4f}"
                elif mtype == "vae":
                    h = None
                    # Report last-epoch losses from training curves
                    curves_fig = res.get("curves_fig")
                    return "Model: VAE — see training curves for loss breakdown."
                return "No metrics available."

            @render.download(filename="pytorch_model.pt")
            async def download_pt_model():
                """Download the trained PyTorch model state dict as a .pt file."""
                res = dl_results()
                if res and res.get("model_bytes"):
                    yield res["model_bytes"]


# ---------------------------------------------------------------------------
# Helper functions — extract outputs from AutoML training_results
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
