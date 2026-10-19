import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import itertools
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, Any, Optional, Tuple, List
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate statistical summary transpose."""
    return df.describe(include='all').transpose()


def generate_target_distribution_plot(df: pd.DataFrame, target: str) -> Optional[Figure]:
    """Generate target variable distribution plot."""
    if target not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x=target, hue=target, palette='viridis', legend=False, ax=ax)
    ax.set_title(f"Class Distribution: {target}")
    fig.tight_layout()
    return fig


def generate_categorical_plots(
    df: pd.DataFrame,
    target: str,
    index_col: Optional[str] = None,
    top_n_categories: int = 20
) -> Dict[str, Figure]:
    """Generate countplots for non-numeric columns."""
    categorical_df = df.select_dtypes(exclude=['number'])
    excluded = [target]
    if index_col:
        excluded.append(index_col)
    cols_to_plot = [col for col in categorical_df.columns if col not in excluded]

    plots = {}
    for col in cols_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        order = df[col].value_counts().index[:top_n_categories]
        sns.countplot(data=df, y=col, hue=col, order=order, palette='magma', legend=False, ax=ax)
        ax.set_title(f"Top Categories: {col}")
        fig.tight_layout()
        plots[col] = fig
    return plots


def generate_correlation_heatmap(
    df: pd.DataFrame,
    index_col: Optional[str] = None
) -> Optional[Figure]:
    """Generate correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        logger.warning("No numeric columns found for correlation.")
        return None

    if index_col and index_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[index_col])

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return fig


def generate_pca_plots(
    df: pd.DataFrame,
    target: str,
    index_col: Optional[str] = None,
    n_components: int = 2,
    impute_strategy: str = "median"
) -> Tuple[Dict[str, Figure], Optional[pd.DataFrame]]:
    """Perform PCA and generate scatter plots for component pairs."""
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        logger.warning("No numeric columns found for PCA.")
        return {}, None

    if index_col and index_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[index_col])

    x = numeric_df.copy()

    # Impute missing values
    if x.isnull().values.any():
        if impute_strategy == 'mean':
            x = x.fillna(x.mean())
        elif impute_strategy == 'median':
            x = x.fillna(x.median())
        elif impute_strategy == 'zero':
            x = x.fillna(0)
        else:
            logger.warning(f"Unknown fill strategy '{impute_strategy}'. Defaulting to median.")
            x = x.fillna(x.median())

    # Standardize and apply PCA
    x_scaled = StandardScaler().fit_transform(x)
    n_comp = min(n_components, x_scaled.shape[1])
    pca = PCA(n_components=n_comp)
    components = pca.fit_transform(x_scaled)

    pca_cols = [f'PC{i+1}' for i in range(n_comp)]
    pca_df = pd.DataFrame(data=components, columns=pca_cols, index=df.index)
    if target in df.columns:
        pca_df[target] = df[target].values

    var_exp = pca.explained_variance_ratio_ * 100
    component_indices = range(1, n_comp + 1)
    plot_pairs = list(itertools.combinations(component_indices, 2))

    pca_plots = {}
    for pc_x, pc_y in plot_pairs:
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            data=pca_df,
            x=f'PC{pc_x}',
            y=f'PC{pc_y}',
            hue=target if target in pca_df.columns else None,
            palette='Set1',
            alpha=0.7,
            ax=ax
        )
        ax.set_title(f"PCA Analysis: PC{pc_x} vs PC{pc_y}")
        ax.set_xlabel(f"PC{pc_x} ({var_exp[pc_x-1]:.2f}% variance)")
        ax.set_ylabel(f"PC{pc_y} ({var_exp[pc_y-1]:.2f}% variance)")
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        pca_plots[f"pca_PC{pc_x}_vs_PC{pc_y}"] = fig

    return pca_plots, pca_df


def run_eda(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Perform Exploratory Data Analysis and save visualizations (CLI Entry)."""
    logger.info("--- Running Exploratory Data Analysis (EDA) ---")

    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_plots = config['output'].get('save_plots', True)

    # 1. Summary
    summary = generate_summary(df)
    summary_path = output_dir / "data_summary.csv"
    summary.to_csv(summary_path)
    logger.info(f"Statistical summary saved to {summary_path}")

    # 2. Target distribution
    target = config['data']['target_column']
    if target in df.columns:
        counts = df[target].value_counts(normalize=True) * 100
        logger.info(f"Class balance (%):\n{counts}")
        if save_plots:
            target_fig = generate_target_distribution_plot(df, target)
            if target_fig:
                path = output_dir / "target_distribution.png"
                target_fig.savefig(path, dpi=150)
                plt.close(target_fig)
                logger.info(f"Target distribution plot saved to {path}")

    # 3. Categorical plots
    index_col = config['data'].get('index_column')
    top_n_cat = config['eda'].get('top_n_categories', 20)
    cat_plots = generate_categorical_plots(df, target, index_col, top_n_cat)
    if save_plots:
        for col, fig in cat_plots.items():
            path = output_dir / f"cat_dist_{col}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            logger.info(f"Categorical plot saved for {col} to {path}")

    # 4. Correlation heatmap
    if save_plots:
        corr_fig = generate_correlation_heatmap(df, index_col)
        if corr_fig:
            path = output_dir / "correlation_heatmap.png"
            corr_fig.savefig(path, dpi=150)
            plt.close(corr_fig)
            logger.info(f"Correlation heatmap saved to {path}")

    # 5. PCA
    if config['eda'].get('generate_pca', False):
        n_comp = config['eda'].get('pca_components', 2)
        impute_strategy = config['preprocessing'].get('impute_missing', 'median')
        pca_plots, _ = generate_pca_plots(df, target, index_col, n_comp, impute_strategy)
        if save_plots:
            for name, fig in pca_plots.items():
                path = output_dir / f"{name}.png"
                fig.savefig(path, dpi=150)
                plt.close(fig)
                logger.info(f"PCA plot saved to {path}")
