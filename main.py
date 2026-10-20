import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.utils import load_config, load_data, drop_unwanted_columns
from src.eda import run_eda
from src.automl_models import run_model_training
from src.bio.rnaseq import RNASeqPipeline
from src.models.dl_expression import (
    ExpressionMLP, ExpressionTrainer,
    ExpressionVAE, VAETrainer,
)

# Initialise logging for the entire application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RNA-Seq pipeline block
# ---------------------------------------------------------------------------

def run_rnaseq_pipeline(config: dict) -> "RNASeqPipeline | None":
    """
    Run the classical RNA-Seq processing pipeline.

    Loads raw counts and metadata, normalises, runs DEA, saves plots and
    CSV results, and optionally performs pathway enrichment.

    Parameters
    ----------
    config : dict
        Full pipeline configuration loaded from config.yaml.

    Returns
    -------
    RNASeqPipeline or None
        The fitted pipeline object (contains self.log_cpm) on success,
        or None if any step fails.
    """
    rna_cfg = config['rnaseq']
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    counts_path   = rna_cfg['count_data_path']
    metadata_path = rna_cfg['metadata_path']
    logger.info(f"RNA-Seq: loading counts from '{counts_path}'.")
    counts   = pd.read_csv(counts_path, index_col=0)
    logger.info(f"RNA-Seq: loading metadata from '{metadata_path}'.")
    metadata = pd.read_csv(metadata_path, index_col=0)

    # --- Instantiate pipeline and normalise ---
    pipe = RNASeqPipeline(
        count_matrix=counts,
        metadata=metadata,
        target_col=rna_cfg['target_col'],
    )
    pipe.normalize_counts(
        min_cpm=rna_cfg.get('min_cpm', 1.0),
        min_samples=rna_cfg.get('min_samples', 3),
    )

    # --- Differential Expression Analysis ---
    dea_results = pipe.run_dea(
        group_a=rna_cfg['group_a'],
        group_b=rna_cfg['group_b'],
        fdr_thresh=rna_cfg.get('fdr_thresh', 0.05),
        log2fc_thresh=rna_cfg.get('log2fc_thresh', 1.0),
    )

    # --- Save DEA results table ---
    dea_csv = output_dir / 'dea_results.csv'
    dea_results.to_csv(dea_csv, index=False)
    logger.info(f"RNA-Seq: DEA results saved to '{dea_csv}'.")

    # --- Save volcano plot ---
    volcano_path = str(output_dir / 'volcano_plot.png')
    pipe.plot_volcano(
        dea_results,
        log2fc_thresh=rna_cfg.get('log2fc_thresh', 1.0),
        fdr_thresh=rna_cfg.get('fdr_thresh', 0.05),
        save_path=volcano_path,
    )
    plt.close('all')

    # --- Save MA plot ---
    ma_path = str(output_dir / 'ma_plot.png')
    pipe.plot_ma(
        dea_results,
        fdr_thresh=rna_cfg.get('fdr_thresh', 0.05),
        save_path=ma_path,
    )
    plt.close('all')

    # --- Optional: pathway enrichment on significant genes ---
    enrichment_db = rna_cfg.get('enrichment_db', '').strip()
    if enrichment_db:
        sig_genes = dea_results.loc[dea_results['significant'], 'gene'].tolist()
        if sig_genes:
            logger.info(
                f"RNA-Seq: running ORA for {len(sig_genes)} significant genes "
                f"against '{enrichment_db}'."
            )
            enrichment_results = pipe.run_pathway_enrichment(
                gene_list=sig_genes,
                gene_sets=enrichment_db,
            )
            if not enrichment_results.empty:
                enr_csv = output_dir / 'pathway_enrichment.csv'
                enrichment_results.to_csv(enr_csv, index=False)
                logger.info(f"RNA-Seq: pathway enrichment saved to '{enr_csv}'.")
        else:
            logger.info("RNA-Seq: no significant genes — skipping pathway enrichment.")

    return pipe


# ---------------------------------------------------------------------------
# Deep Learning pipeline block
# ---------------------------------------------------------------------------

def run_dl_pipeline(config: dict, pipe: "RNASeqPipeline") -> None:
    """
    Train a PyTorch model (MLP or VAE) on the normalised expression matrix.

    Parameters
    ----------
    config : dict
        Full pipeline configuration loaded from config.yaml.
    pipe : RNASeqPipeline
        A fitted RNASeqPipeline whose ``log_cpm`` attribute holds the
        normalised expression matrix.
    """
    dl_cfg     = config['deep_learning']
    rna_cfg    = config['rnaseq']
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_type = dl_cfg.get('model_type', 'mlp').lower()
    log_cpm    = pipe.log_cpm          # DataFrame: samples × genes
    input_dim  = log_cpm.shape[1]
    hidden_dims  = dl_cfg.get('hidden_dims', [512, 256, 128])
    latent_dim   = dl_cfg.get('latent_dim', 32)
    epochs       = dl_cfg.get('epochs', 100)
    lr           = dl_cfg.get('learning_rate', 0.001)
    dropout      = dl_cfg.get('dropout', 0.3)
    batch_size   = dl_cfg.get('batch_size', 32)
    patience     = dl_cfg.get('early_stopping_patience', 10)

    logger.info(f"Deep Learning: model_type='{model_type}', input_dim={input_dim}.")

    # ---- MLP Classifier ------------------------------------------------
    if model_type == 'mlp':
        # Build a binary label from the condition column for supervised training.
        # group_b = 1 (positive class), everything else = 0.
        target_col = rna_cfg['target_col']
        group_b    = rna_cfg['group_b']
        y = (pipe.metadata[target_col] == group_b).astype(int)

        # Align y to log_cpm index order
        y = y.loc[log_cpm.index]
        X_arr = log_cpm.values
        y_arr = y.values

        # Simple 80/20 train / validation split (no shuffle to keep groups intact)
        split = int(len(X_arr) * 0.8)
        X_train, X_val = X_arr[:split], X_arr[split:]
        y_train, y_val = y_arr[:split], y_arr[split:]

        model   = ExpressionMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=2,
            dropout_rate=dropout,
        )
        trainer = ExpressionTrainer(model, task='classification')

        history = trainer.fit(
            X_train, y_train, X_val, y_val,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
        )

        # Save training curve plot
        fig = trainer.plot_training_curves(history)
        curve_path = str(output_dir / 'mlp_training_curves.png')
        fig.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Deep Learning: MLP training curves saved to '{curve_path}'.")

        # Save model checkpoint
        ckpt_path = str(output_dir / 'mlp_model.pt')
        trainer.save_checkpoint(ckpt_path)

    # ---- VAE (Dimensionality Reduction) --------------------------------
    elif model_type == 'vae':
        X_arr = log_cpm.values

        vae     = ExpressionVAE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )
        trainer = VAETrainer(vae)

        history = trainer.fit_vae(
            X_arr,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )

        # Save training curve plot
        fig = trainer.plot_training_curves(history)
        curve_path = str(output_dir / 'vae_training_curves.png')
        fig.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Deep Learning: VAE training curves saved to '{curve_path}'.")

        # Save model checkpoint
        ckpt_path = str(output_dir / 'vae_model.pt')
        trainer.save_checkpoint(ckpt_path)

        # Project mu vectors to 2D with PCA and save scatter plot
        mu_vectors = trainer.transform(X_arr)   # (n_samples, latent_dim)
        pca        = PCA(n_components=2)
        coords_2d  = pca.fit_transform(mu_vectors)

        fig, ax = plt.subplots(figsize=(7, 5))
        target_col = rna_cfg['target_col']
        conditions = pipe.metadata[target_col].values
        unique_conditions = np.unique(conditions)
        colours = plt.cm.tab10(np.linspace(0, 0.5, len(unique_conditions)))

        for cond, colour in zip(unique_conditions, colours):
            mask = conditions == cond
            ax.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                label=cond,
                c=[colour],
                alpha=0.8,
                s=50,
                edgecolors='k',
                linewidths=0.4,
            )

        explained = pca.explained_variance_ratio_ * 100
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}% var)")
        ax.set_title("VAE Latent Space — PCA Projection (mu vectors)")
        ax.legend(title=target_col, framealpha=0.8)
        fig.tight_layout()

        latent_path = str(output_dir / 'vae_latent_space.png')
        fig.savefig(latent_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Deep Learning: VAE latent space plot saved to '{latent_path}'.")

    else:
        logger.warning(
            f"Deep Learning: unknown model_type '{model_type}'. "
            f"Choose 'mlp' or 'vae'. Skipping."
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("--- Initialising bioinformatics pipeline ---")

    # Load configuration
    config = load_config('config.yaml')
    if not config:
        return

    try:
        # ----------------------------------------------------------------
        # 1. Load and preview data (AutoML path)
        # ----------------------------------------------------------------
        df = load_data(config)
        logger.info("Data Preview (first 5 rows):")
        logger.info(f"\n{df.head()}")

        # 2. Drop unwanted columns
        df = drop_unwanted_columns(df, config)

        # 3. Exploratory Data Analysis
        if config['eda']['generate_pca'] or config['output']['save_plots']:
            run_eda(df, config)

        # 4. AutoML model training (sklearn)
        if config.get('models'):
            run_model_training(df, config)

        # ----------------------------------------------------------------
        # 5. RNA-Seq pipeline (gated by rnaseq.enabled)
        # ----------------------------------------------------------------
        rnaseq_pipe = None
        if config.get('rnaseq', {}).get('enabled', False):
            logger.info("--- Running RNA-Seq pipeline ---")
            try:
                rnaseq_pipe = run_rnaseq_pipeline(config)
            except Exception as e:
                logger.error(f"RNA-Seq pipeline failed: {e}")

        # ----------------------------------------------------------------
        # 6. Deep Learning (gated by deep_learning.enabled AND rnaseq ran)
        # ----------------------------------------------------------------
        if config.get('deep_learning', {}).get('enabled', False):
            if rnaseq_pipe is None or rnaseq_pipe.log_cpm is None:
                logger.warning(
                    "Deep Learning is enabled but no normalised expression "
                    "matrix is available. Enable 'rnaseq.enabled: true' first."
                )
            else:
                logger.info("--- Running PyTorch Deep Learning pipeline ---")
                try:
                    run_dl_pipeline(config, rnaseq_pipe)
                except Exception as e:
                    logger.error(f"Deep Learning pipeline failed: {e}")

        logger.info("--- Pipeline execution finished ---")

    except Exception as e:
        logger.error(f"Error occurred during execution: {e}")


if __name__ == "__main__":
    main()
