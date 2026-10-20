"""
Test suite for RNA-Seq (src/bio/rnaseq.py) and PyTorch DL
(src/models/dl_expression.py) modules.

Fast tests run by default.  Training-loop tests are tagged @pytest.mark.slow
and excluded from fast CI:

    pytest tests/test_rnaseq_dl.py -v -m "not slow"
"""

import numpy as np
import pandas as pd
import pytest
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure

from src.bio.rnaseq import RNASeqPipeline
from src.models.dl_expression import (
    ExpressionDataset,
    ExpressionMLP,
    ExpressionTrainer,
    ExpressionVAE,
    VAELoss,
    VAETrainer,
)


# ===========================================================================
# ── RNA-Seq tests ────────────────────────────────────────────────────────────
# ===========================================================================

class TestCPMNormalization:
    """normalize_counts() returns a valid log2(CPM+1) matrix."""

    def test_output_shape_is_valid(self, log_cpm_matrix):
        rows, cols = log_cpm_matrix.shape
        # Genes that survive filtering; with Poisson(50) data essentially all survive.
        assert rows == 20, f"Expected 20 samples, got {rows}"
        assert cols > 0, "All genes were filtered out — check fixture min_cpm/min_samples"

    def test_all_values_non_negative(self, log_cpm_matrix):
        assert (log_cpm_matrix.values >= 0).all(), (
            "log2(CPM+1) values must be ≥ 0"
        )

    def test_stored_on_pipeline(self, count_matrix, metadata):
        pipe = RNASeqPipeline(count_matrix.copy(), metadata.copy(), "condition")
        result = pipe.normalize_counts(min_cpm=0.5, min_samples=2)
        assert pipe.log_cpm is not None
        pd.testing.assert_frame_equal(pipe.log_cpm, result)


class TestDEAOutputStructure:
    """run_dea() returns a DataFrame with the expected schema."""

    REQUIRED_COLUMNS = {"gene", "mean_expr", "log2FC", "p_value", "adj_p_value", "significant"}

    @pytest.fixture(scope="class")
    @classmethod
    def dea_results(cls, count_matrix, metadata):
        pipe = RNASeqPipeline(count_matrix.copy(), metadata.copy(), "condition")
        pipe.normalize_counts(min_cpm=0.5, min_samples=2)
        return pipe.run_dea("Control", "Treated")

    def test_columns_present(self, dea_results):
        assert self.REQUIRED_COLUMNS.issubset(dea_results.columns), (
            f"Missing columns: {self.REQUIRED_COLUMNS - set(dea_results.columns)}"
        )

    def test_p_value_range(self, dea_results):
        assert dea_results["p_value"].between(0, 1).all(), "p_value out of [0, 1]"
        assert dea_results["adj_p_value"].between(0, 1).all(), "adj_p_value out of [0, 1]"

    def test_significant_is_bool(self, dea_results):
        assert dea_results["significant"].dtype == bool, (
            f"'significant' column dtype is {dea_results['significant'].dtype}, expected bool"
        )

    def test_not_empty(self, dea_results):
        assert len(dea_results) > 0, "DEA result is empty"


# ===========================================================================
# ── Plot tests ────────────────────────────────────────────────────────────────
# ===========================================================================

@pytest.fixture(scope="module")
def _pipeline_with_dea(count_matrix, metadata):
    """Return (pipeline, dea_results) ready for plotting."""
    pipe = RNASeqPipeline(count_matrix.copy(), metadata.copy(), "condition")
    pipe.normalize_counts(min_cpm=0.5, min_samples=2)
    dea = pipe.run_dea("Control", "Treated")
    return pipe, dea


class TestVolcanoPlot:
    def test_returns_figure(self, _pipeline_with_dea):
        pipe, dea = _pipeline_with_dea
        fig = pipe.plot_volcano(dea)
        assert isinstance(fig, matplotlib.figure.Figure), (
            f"plot_volcano returned {type(fig)}, expected matplotlib.figure.Figure"
        )

    def test_does_not_call_show(self, _pipeline_with_dea, monkeypatch):
        """Ensure plt.show() is never called inside plot_volcano."""
        import matplotlib.pyplot as plt
        called = []
        monkeypatch.setattr(plt, "show", lambda: called.append(True))
        pipe, dea = _pipeline_with_dea
        pipe.plot_volcano(dea)
        assert not called, "plot_volcano must not call plt.show()"


class TestMAPlot:
    def test_returns_figure(self, _pipeline_with_dea):
        pipe, dea = _pipeline_with_dea
        fig = pipe.plot_ma(dea)
        assert isinstance(fig, matplotlib.figure.Figure), (
            f"plot_ma returned {type(fig)}, expected matplotlib.figure.Figure"
        )

    def test_does_not_call_show(self, _pipeline_with_dea, monkeypatch):
        import matplotlib.pyplot as plt
        called = []
        monkeypatch.setattr(plt, "show", lambda: called.append(True))
        pipe, dea = _pipeline_with_dea
        pipe.plot_ma(dea)
        assert not called, "plot_ma must not call plt.show()"


# ===========================================================================
# ── ExpressionDataset tests ──────────────────────────────────────────────────
# ===========================================================================

class TestExpressionDataset:
    N = 32
    G = 50  # genes / features

    @pytest.fixture
    def dataset(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((self.N, self.G)).astype(np.float32)
        y = rng.integers(0, 2, size=self.N)
        return ExpressionDataset(X, y)

    def test_len(self, dataset):
        assert len(dataset) == self.N

    def test_getitem_types(self, dataset):
        x, y = dataset[0]
        assert x.dtype == torch.float32, f"Feature dtype: {x.dtype}"
        assert y.dtype == torch.float32, f"Target dtype: {y.dtype}"

    def test_getitem_shape(self, dataset):
        x, y = dataset[0]
        assert x.shape == (self.G,), f"Expected ({self.G},), got {x.shape}"
        assert y.shape == (), f"Scalar target expected, got shape {y.shape}"

    def test_accepts_dataframe(self, log_cpm_matrix, metadata):
        """Dataset must work when X is a DataFrame and y is a Series."""
        y = (metadata["condition"] == "Treated").astype(int)
        ds = ExpressionDataset(log_cpm_matrix, y)
        assert len(ds) == len(log_cpm_matrix)
        x0, y0 = ds[0]
        assert x0.dtype == torch.float32


# ===========================================================================
# ── ExpressionMLP tests ──────────────────────────────────────────────────────
# ===========================================================================

class TestExpressionMLP:
    INPUT_DIM = 100
    HIDDEN_DIMS = [64, 32]
    NUM_CLASSES = 2
    BATCH = 8

    @pytest.fixture
    def mlp(self):
        return ExpressionMLP(
            input_dim=self.INPUT_DIM,
            hidden_dims=self.HIDDEN_DIMS,
            num_classes=self.NUM_CLASSES,
            dropout_rate=0.1,
        )

    def test_forward_output_shape(self, mlp):
        x = torch.randn(self.BATCH, self.INPUT_DIM)
        with torch.no_grad():
            out = mlp(x)
        assert out.shape == (self.BATCH, self.NUM_CLASSES), (
            f"Expected ({self.BATCH}, {self.NUM_CLASSES}), got {out.shape}"
        )

    def test_forward_on_cpu(self, mlp):
        mlp = mlp.to("cpu")
        x = torch.randn(self.BATCH, self.INPUT_DIM)
        with torch.no_grad():
            out = mlp(x)
        assert out.device.type == "cpu"

    def test_regression_output(self):
        """num_classes=1 → scalar output per sample."""
        model = ExpressionMLP(50, [32], num_classes=1, dropout_rate=0.0)
        x = torch.randn(4, 50)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"


# ===========================================================================
# ── ExpressionVAE tests ──────────────────────────────────────────────────────
# ===========================================================================

class TestExpressionVAE:
    INPUT_DIM = 100
    HIDDEN_DIMS = [64]
    LATENT_DIM = 8
    BATCH = 8

    @pytest.fixture
    def vae(self):
        return ExpressionVAE(
            input_dim=self.INPUT_DIM,
            hidden_dims=self.HIDDEN_DIMS,
            latent_dim=self.LATENT_DIM,
        )

    def test_forward_shapes(self, vae):
        x = torch.randn(self.BATCH, self.INPUT_DIM)
        with torch.no_grad():
            recon, mu, logvar = vae(x)
        assert recon.shape == (self.BATCH, self.INPUT_DIM), (
            f"recon shape: {recon.shape}"
        )
        assert mu.shape == (self.BATCH, self.LATENT_DIM), (
            f"mu shape: {mu.shape}"
        )
        assert logvar.shape == (self.BATCH, self.LATENT_DIM), (
            f"logvar shape: {logvar.shape}"
        )

    def test_eval_mode_is_deterministic(self, vae):
        """In eval mode reparameterize returns mu → two forward passes identical."""
        vae.eval()
        x = torch.randn(self.BATCH, self.INPUT_DIM)
        with torch.no_grad():
            _, mu1, _ = vae(x)
            _, mu2, _ = vae(x)
        assert torch.equal(mu1, mu2), "mu should be identical across eval forward passes"

    def test_train_mode_samples_latent(self, vae):
        """In train mode reparameterize adds noise → recon_x differs across passes.

        mu itself is deterministic (encoder output); the stochasticity comes
        from the sampled z fed into the decoder, so we compare recon_x.
        """
        vae.train()
        x = torch.randn(self.BATCH, self.INPUT_DIM)
        recon1, _, _ = vae(x)
        recon2, _, _ = vae(x)
        # With high probability the decoded reconstructions differ due to eps sampling
        assert not torch.equal(recon1, recon2), (
            "Decoded reconstructions should differ in train mode (stochastic z)"
        )


class TestVAELoss:
    def test_returns_positive_scalar(self):
        batch, input_dim, latent_dim = 8, 100, 8
        recon_x = torch.randn(batch, input_dim)
        x = torch.randn(batch, input_dim)
        mu = torch.randn(batch, latent_dim)
        logvar = torch.randn(batch, latent_dim)

        criterion = VAELoss()
        loss = criterion(recon_x, x, mu, logvar)

        assert loss.ndim == 0, "VAELoss must return a scalar"
        assert loss.item() > 0, "VAELoss must be positive"

    def test_beta_scales_kl(self):
        """Higher beta → larger loss (KL term weighted more)."""
        batch, idim, ldim = 8, 50, 4
        recon = torch.zeros(batch, idim)
        x = torch.zeros(batch, idim)
        mu = torch.ones(batch, ldim)  # non-zero mu → positive KL
        logvar = torch.zeros(batch, ldim)

        criterion = VAELoss()
        loss_beta1 = criterion(recon, x, mu, logvar, beta=1.0)
        loss_beta5 = criterion(recon, x, mu, logvar, beta=5.0)
        assert loss_beta5.item() > loss_beta1.item(), (
            "Higher beta must increase the VAE loss"
        )


# ===========================================================================
# ── VAETrainer.transform determinism test ────────────────────────────────────
# ===========================================================================

class TestVAETransformDeterminism:
    def test_transform_is_deterministic(self, log_cpm_matrix):
        """Two consecutive transform() calls on the same input return identical arrays."""
        vae = ExpressionVAE(
            input_dim=log_cpm_matrix.shape[1],
            hidden_dims=[32],
            latent_dim=4,
        )
        trainer = VAETrainer(vae, device="cpu")

        z1 = trainer.transform(log_cpm_matrix)
        z2 = trainer.transform(log_cpm_matrix)

        np.testing.assert_array_equal(
            z1, z2,
            err_msg="transform() must return identical arrays (deterministic mu)"
        )

    def test_transform_output_shape(self, log_cpm_matrix):
        latent_dim = 4
        vae = ExpressionVAE(log_cpm_matrix.shape[1], [32], latent_dim)
        trainer = VAETrainer(vae, device="cpu")
        z = trainer.transform(log_cpm_matrix)
        assert z.shape == (len(log_cpm_matrix), latent_dim), (
            f"Expected ({len(log_cpm_matrix)}, {latent_dim}), got {z.shape}"
        )


# ===========================================================================
# ── Slow tests: training loops ───────────────────────────────────────────────
# ===========================================================================

@pytest.mark.slow
class TestExpressionTrainerFit:
    """Full training-loop smoke test — excluded from fast CI."""

    def _split(self, log_cpm_matrix, metadata):
        """Split 20-sample fixture 15/5 train/val."""
        y = (metadata["condition"] == "Treated").astype(int)
        X_train = log_cpm_matrix.iloc[:15]
        X_val   = log_cpm_matrix.iloc[15:]
        y_train = y.iloc[:15]
        y_val   = y.iloc[15:]
        return X_train, X_val, y_train, y_val

    def test_fit_classification(self, log_cpm_matrix, metadata):
        n_genes = log_cpm_matrix.shape[1]
        model = ExpressionMLP(n_genes, [32, 16], num_classes=2, dropout_rate=0.1)
        trainer = ExpressionTrainer(model, task="classification", device="cpu")

        X_tr, X_val, y_tr, y_val = self._split(log_cpm_matrix, metadata)
        history = trainer.fit(
            X_tr, y_tr, X_val, y_val,
            epochs=3,
            lr=1e-3,
            batch_size=4,
            patience=10,
        )

        assert "train_loss" in history, "history must contain 'train_loss'"
        assert len(history["train_loss"]) <= 3, (
            f"Expected ≤3 epochs of loss, got {len(history['train_loss'])}"
        )

    def test_evaluate_returns_metrics(self, log_cpm_matrix, metadata):
        n_genes = log_cpm_matrix.shape[1]
        model = ExpressionMLP(n_genes, [32], num_classes=2, dropout_rate=0.0)
        trainer = ExpressionTrainer(model, task="classification", device="cpu")

        X_tr, X_val, y_tr, y_val = self._split(log_cpm_matrix, metadata)
        trainer.fit(X_tr, y_tr, X_val, y_val, epochs=2, lr=1e-3, batch_size=4)
        metrics = trainer.evaluate(X_val, y_val)

        assert "accuracy" in metrics or "roc_auc" in metrics, (
            f"Expected accuracy/roc_auc in metrics, got keys: {list(metrics.keys())}"
        )

    def test_plot_training_curves_returns_figure(self, log_cpm_matrix, metadata):
        n_genes = log_cpm_matrix.shape[1]
        model = ExpressionMLP(n_genes, [16], num_classes=2, dropout_rate=0.0)
        trainer = ExpressionTrainer(model, task="classification", device="cpu")
        X_tr, X_val, y_tr, y_val = self._split(log_cpm_matrix, metadata)
        history = trainer.fit(X_tr, y_tr, X_val, y_val, epochs=2, lr=1e-3, batch_size=4)

        fig = trainer.plot_training_curves(history)
        assert isinstance(fig, matplotlib.figure.Figure), (
            f"Expected Figure, got {type(fig)}"
        )


@pytest.mark.slow
class TestVAETrainerFit:
    """VAE training-loop smoke test — excluded from fast CI."""

    def test_fit_vae_history_keys(self, log_cpm_matrix):
        n_genes = log_cpm_matrix.shape[1]
        vae = ExpressionVAE(n_genes, [32], latent_dim=4)
        trainer = VAETrainer(vae, device="cpu")

        history = trainer.fit_vae(
            log_cpm_matrix,
            epochs=3,
            lr=1e-3,
            batch_size=4,
        )

        assert "train_loss" in history, "history must contain 'train_loss'"
        assert "recon_loss" in history, "history must contain 'recon_loss'"
        assert "kl_loss" in history, "history must contain 'kl_loss'"
        assert len(history["train_loss"]) == 3, (
            f"Expected 3 epochs, got {len(history['train_loss'])}"
        )

    def test_fit_vae_plot_returns_figure(self, log_cpm_matrix):
        n_genes = log_cpm_matrix.shape[1]
        vae = ExpressionVAE(n_genes, [32], latent_dim=4)
        trainer = VAETrainer(vae, device="cpu")
        history = trainer.fit_vae(log_cpm_matrix, epochs=2, lr=1e-3, batch_size=4)

        fig = trainer.plot_training_curves(history)
        assert isinstance(fig, matplotlib.figure.Figure), (
            f"Expected Figure, got {type(fig)}"
        )
