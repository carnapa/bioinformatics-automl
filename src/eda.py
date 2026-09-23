import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, Any

logger = logging.getLogger(__name__)

def run_eda(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Perform Exploratory Data Analysis and save visualizations."""
    logger.info("--- Running Exploratory Data Analysis (EDA) ---")
    
    # 1.Ensuring output directory exists
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
        
    # 2.Generating and saving statistical summary
    summary = df.describe(include='all').transpose()
    summary_path = output_dir / "data_summary.csv"
    summary.to_csv(summary_path)
    logger.info(f"Statistical summary saved to {summary_path}")

    # 3. Target Variable Balance (Categorical)
    target = config['data']['target_column']
    if target in df.columns:
        plt.figure(figsize=(8, 6))
        # Count values for each class in the target variable
        sns.countplot(data=df, x=target, palette='viridis')
        plt.title(f"Class Distribution: {target}")
        target_dist_path = output_dir / "target_distribution.png"
        if config['output']['save_plots']:
            plt.savefig(target_dist_path)
            logger.info(f"Target distribution plot saved to {target_dist_path}")
        plt.close()
        # Log the balance ratio
        counts = df[target].value_counts(normalize=True) * 100
        logger.info(f"Class balance (%):\n{counts}")

    # 4. Categorical Variables Analysis
    # Select non-numeric columns (strings, categories, objects)
    categorical_df = df.select_dtypes(exclude=['number'])
    # Exclude target and index column
    cols_to_plot = [col for col in categorical_df.columns if col not in [target, config['data']['index_column']]]
    for col in cols_to_plot:
        plt.figure(figsize=(10, 6))
        # Show top N categories if there are too many for any given variable
        top_n_categories = config['eda']['top_n_categories']
        sns.countplot(data=df, y=col, order=df[col].value_counts().index[:top_n_categories], palette='magma')
        plt.title(f"Top Categories: {col}")
        if config['output']['save_plots']:
            cat_plot_path = output_dir / f"cat_dist_{col}.png"
            plt.savefig(cat_plot_path)
            logger.info(f"Categorical plot saved for: {col} to {cat_plot_path}")
        plt.close()

    # 5. Numerical Values Analysis Correlation Heatmap
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        logger.warning("No numeric columns found for correlation or PCA.")
        return

    # Excluding index column if present
    index_col = config['data']['index_column']
    if index_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[index_col])
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    corr_heatmap_path = output_dir / "correlation_heatmap.png"
    # Saving plot based on config preference
    if config['output']['save_plots']:
        plt.savefig(corr_heatmap_path)
        logger.info(f"Correlation heatmap saved to {corr_heatmap_path}")
    plt.close()

    # 6. Principal Component Analysis (PCA)
    if config['eda']['generate_pca']:
        # Copy previous DF with numeric columns and dropped index column if necessary
        x = numeric_df.copy()

        # Handle missing values for PCA filling with median
        if x.isnull().values.any():
            strategy = config['preprocessing'].get('impute_missing', 'median')
            if strategy == 'mean':
                x = x.fillna(x.mean())
            elif strategy == 'median':
                x = x.fillna(x.median())
            elif strategy == 'zero':
                x = x.fillna(0)
            else:
                # Fallback to median if the user types something else
                logger.warning(f"Unknown fill strategy '{strategy}'. Defaulting to median.")
                x = x.fillna(x.median())
            logger.info(f"Missing values imputed using '{strategy}' strategy.")

        # Standardize the data (Mean=0, Variance=1)
        x_scaled = StandardScaler().fit_transform(x)

        # Apply PCA (2 components by default)
        n_comp = config['eda'].get('pca_components',2)
        pca = PCA(n_components=n_comp)
        components = pca.fit_transform(x_scaled)
        
        # Create the Principal Components DataFrame
        pca_cols = [f'PC{i+1}' for i in range(n_comp)]
        pca_df = pd.DataFrame(data=components, columns=pca_cols)
        pca_df[target] = df[target].values # Add target for coloring

        # Calculate explained variance
        var_exp = pca.explained_variance_ratio_ * 100
        
        # Visualization
        component_indices = range(1, n_comp + 1)
        plot_pairs = list(itertools.combinations(component_indices, 2))

        for pc_x, pc_y in plot_pairs:
            plt.figure(figsize=(10, 7))
            sns.scatterplot(
                data=pca_df, 
                x=f'PC{pc_x}', 
                y=f'PC{pc_y}', 
                hue=target, 
                palette='Set1', 
                alpha=0.7
            )
            
            # Using pc_x-1 because the variance array is 0-indexed
            plt.title(f"PCA Analysis: PC{pc_x} vs PC{pc_y}")
            plt.xlabel(f"PC{pc_x} ({var_exp[pc_x-1]:.2f}% variance)")
            plt.ylabel(f"PC{pc_y} ({var_exp[pc_y-1]:.2f}% variance)")
            plt.grid(True, linestyle='--', alpha=0.6)

            # Save PCA plot
            # Saving plot based on config preference
            if config['output']['save_plots']:
                pca_filename = f"pca_PC{pc_x}_vs_PC{pc_y}.png"
                pca_plot_path = output_dir / pca_filename
                plt.savefig(pca_plot_path)
                logger.info(f"PCA plot saved to {pca_plot_path}")
            plt.close()
