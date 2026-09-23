import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, KFold
)
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.inspection import permutation_importance
# Classification models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# Regression models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Default hyperparameter grids
# ---------------------------------------------------------------------------
DEFAULT_CLASSIFICATION_GRIDS = {
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10],
        "max_iter": [1000],
    },
    "svm": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
    },
}
DEFAULT_REGRESSION_GRIDS = {
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0],
    },
    "svr": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
    },
}
# ---------------------------------------------------------------------------
# Model registries
# ---------------------------------------------------------------------------
CLASSIFICATION_MODELS = {
    "random_forest": RandomForestClassifier(random_state=42),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "svm": SVC(probability=True, random_state=42),
    "knn": KNeighborsClassifier(),
}
REGRESSION_MODELS = {
    "random_forest": RandomForestRegressor(random_state=42),
    "ridge": Ridge(),
    "svr": SVR(),
    "knn": KNeighborsRegressor(),
}
# ---------------------------------------------------------------------------
# Task-type detection
# ---------------------------------------------------------------------------
def _detect_task_type(y: pd.Series) -> str:
    """Auto-detect whether the task is classification or regression."""
    if y.dtype == "object" or y.dtype.name == "category":
        return "classification"
    n_unique = y.nunique()
    # Heuristic: few unique values or low cardinality ratio → classification
    if n_unique <= 20 or (n_unique / len(y)) < 0.05:
        return "classification"
    return "regression"
# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _preprocess_data(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series, Optional[LabelEncoder]]:
    """Encode target, impute missing values, apply transforms, and scale features."""
    target_col = config["data"]["target_column"]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # Encode target if categorical
    label_encoder = None
    if y.dtype == "object" or y.dtype.name == "category":
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index, name=target_col)
        mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        logger.info(f"Target '{target_col}' encoded: {mapping}")
    # Keep only numeric features
    X = X.select_dtypes(include=["number"])
    # Impute missing values
    strategy = config["preprocessing"].get("impute_missing", "median")
    if X.isnull().values.any():
        if strategy == "mean":
            X = X.fillna(X.mean())
        elif strategy == "zero":
            X = X.fillna(0)
        else:
            if strategy != "median":
                logger.warning(f"Unknown imputation strategy '{strategy}'. Defaulting to median.")
            X = X.fillna(X.median())
        logger.info(f"Missing values imputed using '{strategy}' strategy.")
    # Optional log transform (applied to strictly positive columns)
    if config["preprocessing"].get("log_transform", False):
        positive_cols = [col for col in X.columns if (X[col] > 0).all()]
        if positive_cols:
            X[positive_cols] = np.log1p(X[positive_cols])
            logger.info(f"Log transform applied to {len(positive_cols)} positive columns.")
    # Scaling
    scaling = config["preprocessing"].get("scaling", "standard")
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    scaler_cls = scalers.get(scaling)
    if scaler_cls is None:
        logger.warning(f"Unknown scaling method '{scaling}'. Defaulting to StandardScaler.")
        scaler_cls = StandardScaler
    X_scaled = pd.DataFrame(
        scaler_cls().fit_transform(X), columns=X.columns, index=X.index
    )
    logger.info(f"Features scaled using '{scaling}' method.")
    return X_scaled, y, label_encoder
# ---------------------------------------------------------------------------
# Model + grid helpers
# ---------------------------------------------------------------------------
def _get_model_and_grid(
    model_name: str, task_type: str, config: Dict[str, Any]
) -> Tuple[Optional[Any], Optional[Dict]]:
    """Return a cloned model instance and its hyperparameter grid."""
    if task_type == "classification":
        base_model = CLASSIFICATION_MODELS.get(model_name)
        default_grid = DEFAULT_CLASSIFICATION_GRIDS.get(model_name, {})
    else:
        base_model = REGRESSION_MODELS.get(model_name)
        default_grid = DEFAULT_REGRESSION_GRIDS.get(model_name, {})
    if base_model is None:
        return None, None
    model = clone(base_model)
    # Allow user-defined grids to override defaults
    user_grids = config.get("model_training", {}).get("hyperparameters", {})
    grid = user_grids.get(model_name, default_grid)
    return model, grid
# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
def _tune_model(
    model, param_grid: Dict, X_train, y_train,
    config: Dict[str, Any], task_type: str
):
    """Optionally tune model with GridSearchCV; otherwise just fit."""
    mt = config.get("model_training", {})
    cv_folds = mt.get("cv_folds", 5)
    use_tuning = mt.get("use_tuning", True)
    if task_type == "classification":
        default_scoring = "f1_weighted"
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        default_scoring = "r2"
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = mt.get("scoring_metric", default_scoring)
    if use_tuning and param_grid:
        logger.info(f"Running GridSearchCV ({cv_folds}-fold, scoring='{scoring}')...")
        gs = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_jobs=-1, verbose=0, refit=True
        )
        gs.fit(X_train, y_train)
        logger.info(f"Best parameters: {gs.best_params_}")
        logger.info(f"Best CV score ({scoring}): {gs.best_score_:.4f}")
        return gs.best_estimator_, gs.best_params_, gs.best_score_
    # No tuning – plain fit
    model.fit(X_train, y_train)
    return model, {}, None
# ---------------------------------------------------------------------------
# Evaluation — Classification
# ---------------------------------------------------------------------------
def _evaluate_classification(
    model, model_name: str, X_test, y_test,
    feature_names: List[str],
    label_encoder: Optional[LabelEncoder],
    output_dir: Path, save_plots: bool
) -> Dict[str, Any]:
    """Compute classification metrics, save report and plots."""
    preds = model.predict(X_test)
    target_names = list(label_encoder.classes_) if label_encoder else None
    # --- Classification report ---
    report = classification_report(y_test, preds, target_names=target_names)
    logger.info(f"Classification Report for {model_name}:\n{report}")
    report_path = output_dir / f"{model_name}_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Classification Report: {model_name}\n{'=' * 50}\n{report}")
    logger.info(f"Report saved to {report_path}")
    # --- Scalar metrics ---
    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, preds, average="weighted", zero_division=0),
    }
    if save_plots:
        # Confusion matrix
        _plot_confusion_matrix(model_name, y_test, preds, target_names, output_dir)
        # ROC curve (binary only)
        auc_val = _plot_roc_curve(model, model_name, X_test, y_test, output_dir)
        if auc_val is not None:
            metrics["auc"] = auc_val
        # Feature importance
        _plot_feature_importance(model, model_name, X_test, y_test, feature_names, output_dir)
    return metrics
def _plot_confusion_matrix(
    model_name: str, y_test, preds, target_names, output_dir: Path
) -> None:
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names
    )
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    path = output_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {path}")
def _plot_roc_curve(
    model, model_name: str, X_test, y_test, output_dir: Path
) -> Optional[float]:
    """Plot ROC curve for binary classification. Returns AUC or None."""
    if len(np.unique(y_test)) != 2:
        return None
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            return None
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {model_name}")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        path = output_dir / f"{model_name}_roc_curve.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"ROC curve saved to {path}")
        return roc_auc
    except Exception as e:
        logger.warning(f"Could not generate ROC curve for {model_name}: {e}")
        return None
# ---------------------------------------------------------------------------
# Evaluation — Regression
# ---------------------------------------------------------------------------
def _evaluate_regression(
    model, model_name: str, X_test, y_test,
    feature_names: List[str],
    output_dir: Path, save_plots: bool
) -> Dict[str, Any]:
    """Compute regression metrics, save report and plots."""
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    logger.info(
        f"Regression Metrics for {model_name}: "
        f"R²={r2:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}"
    )
    metrics: Dict[str, Any] = {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae}
    # Save text report
    report_path = output_dir / f"{model_name}_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Regression Report: {model_name}\n{'=' * 50}\n")
        f.write(f"R²:   {r2:.4f}\nMSE:  {mse:.4f}\nRMSE: {rmse:.4f}\nMAE:  {mae:.4f}\n")
    logger.info(f"Report saved to {report_path}")
    if save_plots:
        _plot_predicted_vs_actual(model_name, y_test, preds, r2, output_dir)
        _plot_residuals(model_name, y_test, preds, output_dir)
        _plot_feature_importance(model, model_name, X_test, y_test, feature_names, output_dir)
    return metrics
def _plot_predicted_vs_actual(
    model_name: str, y_test, preds, r2: float, output_dir: Path
) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, alpha=0.6, color="steelblue", edgecolors="k", linewidth=0.5)
    bounds = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    plt.plot(bounds, bounds, "r--", lw=2, label="Perfect prediction")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual: {model_name} (R²={r2:.4f})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    path = output_dir / f"{model_name}_predicted_vs_actual.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Predicted vs Actual plot saved to {path}")
def _plot_residuals(
    model_name: str, y_test, preds, output_dir: Path
) -> None:
    residuals = y_test - preds
    plt.figure(figsize=(8, 6))
    plt.scatter(preds, residuals, alpha=0.6, color="coral", edgecolors="k", linewidth=0.5)
    plt.axhline(y=0, color="navy", linestyle="--", lw=2)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"Residuals Plot: {model_name}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    path = output_dir / f"{model_name}_residuals.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Residuals plot saved to {path}")
# ---------------------------------------------------------------------------
# Feature importance (shared by classification and regression)
# ---------------------------------------------------------------------------
def _plot_feature_importance(
    model, model_name: str, X_test, y_test,
    feature_names: List[str], output_dir: Path, top_n: int = 20
) -> None:
    """Plot native or permutation-based feature importance."""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            importance_label = "Feature Importance"
        elif hasattr(model, "coef_"):
            coef = np.atleast_2d(model.coef_)
            importances = np.abs(coef).mean(axis=0)
            importance_label = "Coefficient Magnitude"
        else:
            result = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
            )
            importances = result.importances_mean
            importance_label = "Permutation Importance"
        indices = np.argsort(importances)[::-1][:top_n]
        top_feat = [feature_names[i] for i in indices]
        top_imp = importances[indices]
        plt.figure(figsize=(10, max(6, len(top_feat) * 0.3)))
        plt.barh(range(len(top_feat)), top_imp[::-1], color="teal", edgecolor="k", linewidth=0.5)
        plt.yticks(range(len(top_feat)), top_feat[::-1])
        plt.xlabel(importance_label)
        plt.title(f"{importance_label}: {model_name} (Top {len(top_feat)})")
        plt.tight_layout()
        path = output_dir / f"{model_name}_feature_importance.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Feature importance plot saved to {path}")
    except Exception as e:
        logger.warning(f"Could not compute feature importance for {model_name}: {e}")
# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def run_model_training(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Train and evaluate all models specified in config."""
    logger.info("--- Running Model Training ---")
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_plots = config["output"].get("save_plots", True)
    save_models = config["output"].get("save_models", True)
    # Preprocess
    X, y, label_encoder = _preprocess_data(df, config)
    feature_names = list(X.columns)
    # Detect task type
    mt = config.get("model_training", {})
    task_type = mt.get("task_type", "auto")
    if task_type == "auto":
        task_type = _detect_task_type(df[config["data"]["target_column"]])
    logger.info(f"Detected task type: {task_type}")
    # Train / test split
    test_size = config["preprocessing"].get("test_size", 0.2)
    split_kwargs = dict(test_size=test_size, random_state=42)
    if task_type == "classification":
        split_kwargs["stratify"] = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
    logger.info(f"Data split: train={X_train.shape[0]}, test={X_test.shape[0]}")
    # Iterate over models
    selected_models = config.get("models", [])
    all_results: List[Dict[str, Any]] = []
    for model_name in selected_models:
        logger.info(f"{'=' * 50}")
        logger.info(f"--- Training Model: {model_name} ---")
        model, param_grid = _get_model_and_grid(model_name, task_type, config)
        if model is None:
            logger.warning(
                f"Model '{model_name}' is not available for task type "
                f"'{task_type}'. Skipping."
            )
            continue
        try:
            best_model, best_params, best_cv = _tune_model(
                model, param_grid, X_train, y_train, config, task_type
            )
            if task_type == "classification":
                metrics = _evaluate_classification(
                    best_model, model_name, X_test, y_test,
                    feature_names, label_encoder, output_dir, save_plots
                )
            else:
                metrics = _evaluate_regression(
                    best_model, model_name, X_test, y_test,
                    feature_names, output_dir, save_plots
                )
            result = {
                "model": model_name,
                "task_type": task_type,
                **metrics,
                "best_params": str(best_params),
                "best_cv_score": best_cv,
            }
            all_results.append(result)
            if save_models:
                model_path = output_dir / f"{model_name}_model.joblib"
                joblib.dump(best_model, model_path)
                logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
    # Save comparison summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        comparison_path = output_dir / "model_comparison.csv"
        results_df.to_csv(comparison_path, index=False)
        logger.info(f"Model comparison saved to {comparison_path}")
        logger.info(f"\nModel Comparison:\n{results_df.to_string(index=False)}")
    logger.info("--- Model Training Complete ---")