"""
src/bio/rnaseq.py
=================
Classical RNA-Seq processing pipeline.

Provides the RNASeqPipeline class for:
  - CPM normalisation and log2(CPM + 1) transformation
  - Low-expression gene filtering
  - Differential Expression Analysis (DEA) via Welch's t-test or Mann-Whitney U
  - Benjamini-Hochberg FDR correction
  - Volcano and MA plot generation (added in Step 3)
  - Over-Representation Analysis (ORA) via gseapy (added in Step 3)

Usage:
    from src.bio.rnaseq import RNASeqPipeline
"""

import logging
import warnings
from pathlib import Path
from typing import Literal, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for headless / server use
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


class RNASeqPipeline:
    """
    End-to-end classical RNA-Seq analysis pipeline.

    Handles normalisation, filtering, differential expression analysis,
    visualisation, and pathway enrichment for raw gene count matrices.

    Parameters
    ----------
    count_matrix : pd.DataFrame
        Raw read counts. Convention: samples as ROWS, genes as COLUMNS.
        If the matrix has more columns than rows (i.e. it appears to be
        genes × samples), it is automatically transposed with a warning.
    metadata : pd.DataFrame
        Sample metadata. Its index must match the count matrix index
        (sample IDs). Must contain at least the column specified by
        ``target_col``.
    target_col : str
        Column in ``metadata`` that holds the condition/group labels
        (e.g. "condition", "treatment").
    """

    def __init__(
        self,
        count_matrix: pd.DataFrame,
        metadata: pd.DataFrame,
        target_col: str,
    ) -> None:
        # --- Orientation detection: enforce samples × genes layout ---
        # Strategy: check whether the matrix index or the matrix columns
        # align with the metadata index (sample IDs).
        #   - If the index already matches → correct orientation, nothing to do.
        #   - If the columns match instead → matrix is genes × samples; transpose.
        #   - If neither matches → raise a clear ValueError.
        meta_ids = set(metadata.index)
        index_match = set(count_matrix.index) == meta_ids
        column_match = set(count_matrix.columns) == meta_ids

        if index_match:
            # Already in samples × genes orientation — nothing to do.
            pass
        elif column_match:
            logger.warning(
                "Count matrix appears to be genes × samples "
                f"(shape {count_matrix.shape}). Transposing to samples × genes."
            )
            count_matrix = count_matrix.T
        else:
            # Neither orientation aligns with the metadata sample IDs.
            count_ids = set(count_matrix.index)
            only_counts = count_ids - meta_ids
            only_meta = meta_ids - count_ids
            msg_parts = []
            if only_counts:
                msg_parts.append(
                    f"in count matrix index but not metadata: {sorted(only_counts)[:10]}..."
                )
            if only_meta:
                msg_parts.append(
                    f"in metadata but not count matrix: {sorted(only_meta)[:10]}..."
                )
            raise ValueError(
                "Sample IDs do not match between count matrix and metadata.\n"
                "Checked both index and columns of the count matrix.\n"
                + "\n".join(msg_parts)
            )

        # --- Validate target column exists ---
        if target_col not in metadata.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in metadata columns: "
                f"{list(metadata.columns)}"
            )

        # Align row order: metadata rows follow the count matrix row order
        self.counts: pd.DataFrame = count_matrix
        self.metadata: pd.DataFrame = metadata.loc[count_matrix.index]
        self.target_col: str = target_col

        # Populated by normalize_counts()
        self.log_cpm: pd.DataFrame | None = None

        logger.info(
            f"RNASeqPipeline initialised: {self.counts.shape[0]} samples, "
            f"{self.counts.shape[1]} genes."
        )

    # ------------------------------------------------------------------
    # 1. Normalisation & filtering
    # ------------------------------------------------------------------

    def normalize_counts(
        self,
        min_cpm: float = 1.0,
        min_samples: int = 3,
    ) -> pd.DataFrame:
        """
        Compute CPM, filter low-expression genes, and return log2(CPM + 1).

        Parameters
        ----------
        min_cpm : float
            Minimum CPM value a gene must reach in at least ``min_samples``
            samples to be retained. Default: 1.0.
        min_samples : int
            Minimum number of samples in which a gene must meet ``min_cpm``.
            Default: 3.

        Returns
        -------
        pd.DataFrame
            log2(CPM + 1) expression matrix (samples × genes, filtered).
            Also stored as ``self.log_cpm``.
        """
        logger.info("Computing CPM normalisation...")

        # Library size = total counts per sample (row sum)
        library_sizes = self.counts.sum(axis=1)  # shape: (n_samples,)

        # CPM: divide each row by its library size (in millions)
        # Broadcasting: divide each row element-wise by the row's library size
        cpm = self.counts.div(library_sizes, axis=0) * 1e6

        # --- Low-expression filter ---
        genes_before = cpm.shape[1]
        # A gene passes if it meets min_cpm in at least min_samples samples
        passes_filter = (cpm >= min_cpm).sum(axis=0) >= min_samples
        cpm_filtered = cpm.loc[:, passes_filter]
        genes_after = cpm_filtered.shape[1]
        genes_removed = genes_before - genes_after

        logger.info(
            f"Gene filtering: {genes_before} → {genes_after} genes retained "
            f"(removed {genes_removed} genes with CPM < {min_cpm} in fewer "
            f"than {min_samples} samples)."
        )

        # --- log2(CPM + 1) transformation ---
        log_cpm = np.log2(cpm_filtered + 1)

        self.log_cpm = log_cpm
        logger.info(
            f"Normalisation complete. log_cpm matrix shape: {self.log_cpm.shape}"
        )
        return self.log_cpm

    # ------------------------------------------------------------------
    # 2. Differential Expression Analysis
    # ------------------------------------------------------------------

    def run_dea(
        self,
        group_a: str,
        group_b: str,
        method: Literal["ttest", "mannwhitney"] = "ttest",
        fdr_thresh: float = 0.05,
        log2fc_thresh: float = 1.0,
    ) -> pd.DataFrame:
        """
        Perform Differential Expression Analysis between two sample groups.

        Computes per-gene log2 Fold Change, raw p-values, and
        Benjamini-Hochberg adjusted p-values.

        Parameters
        ----------
        group_a : str
            Reference group label in ``metadata[target_col]`` (denominator
            of the fold change).
        group_b : str
            Comparison group label (numerator of the fold change).
        method : {'ttest', 'mannwhitney'}
            Statistical test to use:
            - 'ttest'       : Welch's t-test (scipy.stats.ttest_ind,
                              equal_var=False). Assumes approximate normality.
            - 'mannwhitney' : Mann-Whitney U test (non-parametric).
            Default: 'ttest'.
        fdr_thresh : float
            Adjusted p-value threshold for the ``significant`` flag.
            Default: 0.05.
        log2fc_thresh : float
            Absolute log2FC threshold for the ``significant`` flag.
            Default: 1.0.

        Returns
        -------
        pd.DataFrame
            Results table with columns:
            ``[gene, mean_expr, log2FC, p_value, adj_p_value, significant]``
            sorted by ``adj_p_value`` ascending.

        Raises
        ------
        RuntimeError
            If ``normalize_counts()`` has not been called first.
        ValueError
            If ``group_a`` or ``group_b`` are not found in ``target_col``.
        """
        if self.log_cpm is None:
            raise RuntimeError(
                "normalize_counts() must be called before run_dea()."
            )

        # --- Validate group labels ---
        available_groups = self.metadata[self.target_col].unique().tolist()
        for label, grp in [("group_a", group_a), ("group_b", group_b)]:
            if grp not in available_groups:
                raise ValueError(
                    f"{label}='{grp}' not found in metadata['{self.target_col}']. "
                    f"Available: {available_groups}"
                )

        logger.info(
            f"Running DEA: '{group_a}' vs '{group_b}' using method='{method}' "
            f"(FDR ≤ {fdr_thresh}, |log2FC| ≥ {log2fc_thresh})."
        )

        # --- Subset samples per group ---
        samples_a = self.metadata.index[
            self.metadata[self.target_col] == group_a
        ]
        samples_b = self.metadata.index[
            self.metadata[self.target_col] == group_b
        ]

        expr_a = self.log_cpm.loc[samples_a]  # shape: (n_a, n_genes)
        expr_b = self.log_cpm.loc[samples_b]  # shape: (n_b, n_genes)

        logger.info(
            f"Group sizes — '{group_a}': {len(samples_a)} samples, "
            f"'{group_b}': {len(samples_b)} samples."
        )

        genes = self.log_cpm.columns.tolist()
        n_genes = len(genes)

        # --- Per-gene statistics ---
        mean_a = expr_a.mean(axis=0).values   # shape: (n_genes,)
        mean_b = expr_b.mean(axis=0).values

        # log2FC = mean(group_b) − mean(group_a) in log2 space
        log2fc = mean_b - mean_a

        # Mean expression across both groups combined
        mean_expr = self.log_cpm.loc[
            list(samples_a) + list(samples_b)
        ].mean(axis=0).values

        # --- Raw p-values ---
        p_values = np.empty(n_genes, dtype=float)

        if method == "ttest":
            for i, gene in enumerate(genes):
                _, p_values[i] = stats.ttest_ind(
                    expr_b[gene].values,
                    expr_a[gene].values,
                    equal_var=False,   # Welch's t-test
                )
        elif method == "mannwhitney":
            for i, gene in enumerate(genes):
                _, p_values[i] = stats.mannwhitneyu(
                    expr_b[gene].values,
                    expr_a[gene].values,
                    alternative="two-sided",
                )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'ttest' or 'mannwhitney'."
            )

        # Replace any NaN p-values (e.g. zero-variance genes) with 1.0
        p_values = np.where(np.isnan(p_values), 1.0, p_values)

        # --- Benjamini-Hochberg FDR correction ---
        _, adj_p_values, _, _ = multipletests(
            p_values, alpha=fdr_thresh, method="fdr_bh"
        )

        # --- Significance flag ---
        significant = (adj_p_values < fdr_thresh) & (np.abs(log2fc) >= log2fc_thresh)

        # --- Assemble results DataFrame ---
        results = pd.DataFrame(
            {
                "gene":        genes,
                "mean_expr":   mean_expr,
                "log2FC":      log2fc,
                "p_value":     p_values,
                "adj_p_value": adj_p_values,
                "significant": significant.astype(bool),
            }
        ).sort_values("adj_p_value").reset_index(drop=True)

        n_sig = results["significant"].sum()
        logger.info(
            f"DEA complete: {n_sig} / {n_genes} genes are significant "
            f"(FDR ≤ {fdr_thresh} & |log2FC| ≥ {log2fc_thresh})."
        )

        return results

    # ------------------------------------------------------------------
    # 3. Visualisation — Volcano Plot
    # ------------------------------------------------------------------

    def plot_volcano(
        self,
        dea_results: pd.DataFrame,
        log2fc_thresh: float = 1.0,
        fdr_thresh: float = 0.05,
        save_path: Optional[str] = None,
    ) -> matplotlib.figure.Figure:
        """
        Generate a volcano plot from DEA results.

        Plots −log10(adj_p_value) on the y-axis against log2FC on the
        x-axis.  Genes are coloured by regulation direction:
        - **Red**  — up-regulated   (log2FC ≥ threshold  AND FDR significant)
        - **Blue** — down-regulated (log2FC ≤ −threshold AND FDR significant)
        - **Grey** — not significant

        Dashed lines mark the log2FC and −log10(FDR) thresholds.

        Parameters
        ----------
        dea_results : pd.DataFrame
            Output of :meth:`run_dea` containing columns
            ``[gene, log2FC, adj_p_value, significant]``.
        log2fc_thresh : float
            Absolute log2FC threshold for the dashed vertical lines.
            Default: 1.0.
        fdr_thresh : float
            Adjusted p-value threshold for the dashed horizontal line.
            Default: 0.05.
        save_path : str, optional
            If provided, the figure is saved to this file path (PNG/SVG/PDF
            depending on extension). The Figure is still returned regardless.

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = dea_results.copy()

        # Avoid log10(0) by clipping adj_p_value to a small positive floor
        min_nonzero = df.loc[df["adj_p_value"] > 0, "adj_p_value"].min()
        floor = min_nonzero / 10 if pd.notna(min_nonzero) else 1e-300
        df["adj_p_value_clipped"] = df["adj_p_value"].clip(lower=floor)
        df["neg_log10_adj_p"] = -np.log10(df["adj_p_value_clipped"])

        # Assign colour category per gene
        sig_thresh = -np.log10(fdr_thresh)

        def _colour_category(row: pd.Series) -> str:
            if row["adj_p_value"] < fdr_thresh and row["log2FC"] >= log2fc_thresh:
                return "Up-regulated"
            elif row["adj_p_value"] < fdr_thresh and row["log2FC"] <= -log2fc_thresh:
                return "Down-regulated"
            return "Not significant"

        df["regulation"] = df.apply(_colour_category, axis=1)

        palette = {
            "Up-regulated":   "#E41A1C",   # red
            "Down-regulated": "#377EB8",   # blue
            "Not significant": "#AAAAAA",  # grey
        }
        order = ["Not significant", "Up-regulated", "Down-regulated"]

        fig, ax = plt.subplots(figsize=(9, 6))

        for category in order:
            subset = df[df["regulation"] == category]
            ax.scatter(
                subset["log2FC"],
                subset["neg_log10_adj_p"],
                c=palette[category],
                label=f"{category} (n={len(subset)})",
                alpha=0.6,
                s=18,
                linewidths=0,
            )

        # Threshold lines
        ax.axhline(
            y=sig_thresh, color="black", linestyle="--",
            linewidth=0.8, label=f"FDR = {fdr_thresh}"
        )
        ax.axvline(
            x=log2fc_thresh, color="black", linestyle=":",
            linewidth=0.8, label=f"|log2FC| = {log2fc_thresh}"
        )
        ax.axvline(x=-log2fc_thresh, color="black", linestyle=":", linewidth=0.8)

        # Labels and formatting
        ax.set_xlabel("log₂ Fold Change", fontsize=12)
        ax.set_ylabel("−log₁₀ (Adjusted p-value)", fontsize=12)
        ax.set_title("Volcano Plot — Differential Expression Analysis", fontsize=13)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
        sns.despine(ax=ax)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Volcano plot saved to '{save_path}'.")

        return fig

    # ------------------------------------------------------------------
    # 4. Visualisation — MA Plot
    # ------------------------------------------------------------------

    def plot_ma(
        self,
        dea_results: pd.DataFrame,
        fdr_thresh: float = 0.05,
        save_path: Optional[str] = None,
    ) -> matplotlib.figure.Figure:
        """
        Generate an MA plot (Bland-Altman style) from DEA results.

        Plots log2FC (M — the "ratio") on the y-axis against mean expression
        (A — the "average") on the x-axis.  Significant genes are coloured
        red; non-significant genes are grey.  A horizontal reference line
        at y = 0 marks no differential expression.

        Parameters
        ----------
        dea_results : pd.DataFrame
            Output of :meth:`run_dea` containing columns
            ``[gene, mean_expr, log2FC, significant]``.
        fdr_thresh : float
            Adjusted p-value threshold used to label the legend correctly.
            Default: 0.05.
        save_path : str, optional
            If provided, the figure is saved to this path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = dea_results.copy()

        sig_mask = df["significant"].astype(bool)
        n_sig = sig_mask.sum()
        n_nonsig = (~sig_mask).sum()

        fig, ax = plt.subplots(figsize=(9, 6))

        # Non-significant genes — drawn first (background layer)
        ax.scatter(
            df.loc[~sig_mask, "mean_expr"],
            df.loc[~sig_mask, "log2FC"],
            c="#AAAAAA",
            alpha=0.5,
            s=14,
            linewidths=0,
            label=f"Not significant (n={n_nonsig})",
        )

        # Significant genes — drawn on top
        ax.scatter(
            df.loc[sig_mask, "mean_expr"],
            df.loc[sig_mask, "log2FC"],
            c="#E41A1C",
            alpha=0.7,
            s=18,
            linewidths=0,
            label=f"Significant (FDR < {fdr_thresh}, n={n_sig})",
        )

        # Reference line at M = 0 (no differential expression)
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.9)

        # Labels and formatting
        ax.set_xlabel("Mean Expression (log₂ CPM + 1)", fontsize=12)
        ax.set_ylabel("log₂ Fold Change", fontsize=12)
        ax.set_title("MA Plot — Mean Expression vs. Fold Change", fontsize=13)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
        sns.despine(ax=ax)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"MA plot saved to '{save_path}'.")

        return fig

    # ------------------------------------------------------------------
    # 5. Pathway Enrichment — Over-Representation Analysis (ORA)
    # ------------------------------------------------------------------

    def run_pathway_enrichment(
        self,
        gene_list: list,
        gene_sets: str = "KEGG_2021_Human",
        gene_sets_local: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Perform Over-Representation Analysis (ORA) using Enrichr via gseapy.

        Queries either the Enrichr online database or a local ``.gmt`` file
        for pathway enrichment of the supplied gene list.  Network errors are
        caught gracefully and an empty result DataFrame is returned with a
        warning rather than raising an exception.

        Parameters
        ----------
        gene_list : list of str
            Gene symbols (or IDs) to test for enrichment — typically the
            significant genes from :meth:`run_dea`.
        gene_sets : str
            Enrichr library name to query online (e.g. ``'KEGG_2021_Human'``,
            ``'GO_Biological_Process_2023'``).  Ignored when
            ``gene_sets_local`` is set.  Default: ``'KEGG_2021_Human'``.
        gene_sets_local : str, optional
            Path to a local ``.gmt`` gene-set file.  When provided, gseapy
            uses this file instead of querying the Enrichr API, enabling
            fully offline analysis.

        Returns
        -------
        pd.DataFrame
            Enrichment results with columns:
            ``[Gene_set, Term, P-value, Adjusted P-value, Genes]``.
            Returns an empty DataFrame with those columns on failure.
        """
        # Column schema guaranteed in both success and failure paths
        _empty_cols = ["Gene_set", "Term", "P-value", "Adjusted P-value", "Genes"]

        if not gene_list:
            logger.warning("run_pathway_enrichment: gene_list is empty — skipping.")
            return pd.DataFrame(columns=_empty_cols)

        # Determine the gene-set source to use
        source = gene_sets_local if gene_sets_local else gene_sets
        logger.info(
            f"Running ORA enrichment for {len(gene_list)} genes "
            f"against gene sets: '{source}'."
        )

        try:
            import gseapy as gp

            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=source,
                outdir=None,      # Do not write files to disk
                verbose=False,
            )
            results: pd.DataFrame = enr.results

            if results.empty:
                logger.info("Enrichment analysis returned no results.")
                return pd.DataFrame(columns=_empty_cols)

            # Standardise column names to the agreed schema
            # gseapy returns: Gene_set, Term, Overlap, P-value, Adjusted P-value,
            #                 Old P-value, Old Adjusted P-value, Odds Ratio,
            #                 Combined Score, Genes
            keep = [c for c in _empty_cols if c in results.columns]
            results = results[keep].reset_index(drop=True)

            logger.info(
                f"ORA complete: {len(results)} enriched terms returned."
            )
            return results

        except (ConnectionError, TimeoutError) as net_err:
            logger.warning(
                f"Network error during pathway enrichment — "
                f"returning empty results. Details: {net_err}"
            )
        except Exception as err:
            # Catch requests-level errors (e.g. requests.exceptions.ConnectionError)
            # and any unexpected gseapy failures without crashing the pipeline.
            err_type = type(err).__name__
            err_module = type(err).__module__
            if "requests" in err_module or "urllib" in err_module or "http" in err_module:
                logger.warning(
                    f"HTTP/requests error during pathway enrichment — "
                    f"returning empty results. Details: {err}"
                )
            else:
                logger.warning(
                    f"Unexpected error ({err_type}) during pathway enrichment — "
                    f"returning empty results. Details: {err}"
                )

        return pd.DataFrame(columns=_empty_cols)
