"""
Shared pytest fixtures for the RNA-Seq and deep-learning test suite.

Fixture design:
  - count_matrix : (20 samples × 100 genes) Poisson-distributed raw counts.
    Genes are COLUMNS, samples are ROWS (sklearn convention).
  - metadata     : matching index (S01…S20), single 'condition' column
    with 10 "Control" + 10 "Treated" labels.
  - log_cpm_matrix : result of RNASeqPipeline.normalize_counts() run on
    count_matrix / metadata.  Stored once and reused across tests.
"""

import numpy as np
import pandas as pd
import pytest

from src.bio.rnaseq import RNASeqPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
N_SAMPLES = 20
N_GENES = 100
RNG_SEED = 42


def _sample_ids() -> list[str]:
    return [f"S{i:02d}" for i in range(1, N_SAMPLES + 1)]


def _gene_ids() -> list[str]:
    return [f"GENE_{i:03d}" for i in range(1, N_GENES + 1)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def count_matrix() -> pd.DataFrame:
    """Raw count matrix: 20 samples × 100 genes, Poisson(lam=50)."""
    rng = np.random.default_rng(RNG_SEED)
    data = rng.poisson(lam=50, size=(N_SAMPLES, N_GENES))
    return pd.DataFrame(data, index=_sample_ids(), columns=_gene_ids())


@pytest.fixture(scope="session")
def metadata() -> pd.DataFrame:
    """Sample metadata: same index as count_matrix, 'condition' column."""
    conditions = ["Control"] * 10 + ["Treated"] * 10
    return pd.DataFrame(
        {"condition": conditions},
        index=_sample_ids(),
    )


@pytest.fixture(scope="session")
def log_cpm_matrix(count_matrix, metadata) -> pd.DataFrame:
    """
    log2(CPM+1) matrix produced by RNASeqPipeline.normalize_counts().

    Uses relaxed min_samples=2 so genes survive the Poisson(50) filter
    even with the small (20-sample) fixture matrix.
    """
    pipeline = RNASeqPipeline(count_matrix.copy(), metadata.copy(), "condition")
    return pipeline.normalize_counts(min_cpm=0.5, min_samples=2)
