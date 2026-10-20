"""
src/models/dl_expression.py
============================
PyTorch deep learning models for high-dimensional gene expression data.

Provides:
  - ExpressionDataset  : PyTorch Dataset wrapping log2(CPM+1) matrices
  - ExpressionMLP      : Configurable Multi-Layer Perceptron classifier/regressor
  - ExpressionTrainer  : Training loop with early stopping for supervised models
  - ExpressionVAE      : Variational Autoencoder for non-linear dim. reduction
                         (added in Step 5)
  - VAELoss            : Combined MSE + KL-divergence loss with beta annealing
                         (added in Step 5)
  - VAETrainer         : Training loop for the VAE with KL annealing schedule
                         (added in Step 5)

Usage:
    from src.models.dl_expression import ExpressionMLP, ExpressionTrainer
    from src.models.dl_expression import ExpressionVAE, VAETrainer
"""

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for headless / server use
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: coerce various input types to a 2-D float32 numpy array
# ---------------------------------------------------------------------------

def _to_numpy(X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
    """
    Convert a DataFrame, Series, or ndarray to a float32 numpy array.

    DataFrames and Series have their index reset before conversion to avoid
    shape surprises when the index carries sample IDs (e.g. 'S01', 'S02', ...).
    """
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True).values.astype(np.float32)
    if isinstance(X, pd.Series):
        return X.reset_index(drop=True).values.astype(np.float32)
    return np.asarray(X, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

class ExpressionDataset(Dataset):
    """
    PyTorch Dataset for gene expression matrices.

    Wraps a log2(CPM+1) expression matrix and its associated target labels
    into a format compatible with PyTorch DataLoaders.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_genes)
        Expression values — numpy array or pandas DataFrame.
        DataFrames have their index reset automatically.
    y : array-like, shape (n_samples,)
        Target labels (class integers for classification, floats for
        regression) — numpy array, pandas Series, or list.
    """

    def __init__(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, list],
    ) -> None:
        X_arr = _to_numpy(X)
        y_arr = _to_numpy(y).ravel()   # ensure 1-D

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Got X.shape={X_arr.shape}, y.shape={y_arr.shape}."
            )

        # Store as float32 tensors; targets cast to long inside trainer for
        # CrossEntropyLoss when needed
        self.X = torch.from_numpy(X_arr)   # (n_samples, n_genes)
        self.y = torch.from_numpy(y_arr)   # (n_samples,)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# 2. Model — Multi-Layer Perceptron
# ---------------------------------------------------------------------------

class ExpressionMLP(nn.Module):
    """
    Flexible Multi-Layer Perceptron for gene expression classification or
    regression.

    Each hidden block follows the pattern:
        Linear -> BatchNorm1d -> ReLU -> Dropout

    This regularisation scheme (BatchNorm before activation, Dropout last) is
    well-suited to high-dimensional, low-sample-count data (p >> n), which is
    the typical regime for bulk RNA-Seq expression matrices.

    Parameters
    ----------
    input_dim : int
        Number of input features (genes after filtering).
    hidden_dims : list of int
        Sizes of each hidden layer in order, e.g. ``[512, 256, 128]``.
    num_classes : int
        Output dimensionality.  Use the number of classes for classification
        or ``1`` for regression.
    dropout_rate : float
        Dropout probability applied after each hidden block. Default: 0.3.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size.")

        # Build hidden blocks dynamically from the hidden_dims list
        layers: List[nn.Module] = []
        in_features = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            ])
            in_features = h_dim

        self.hidden = nn.Sequential(*layers)

        # Output layer — no activation applied here; the caller or loss
        # function handles softmax / sigmoid as appropriate
        self.output = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all hidden blocks and the output layer."""
        x = self.hidden(x)
        return self.output(x)


# ---------------------------------------------------------------------------
# 3. Trainer — supervised (classification & regression)
# ---------------------------------------------------------------------------

class ExpressionTrainer:
    """
    Supervised training loop for :class:`ExpressionMLP` (or any nn.Module).

    Supports both classification and regression, implements early stopping
    with best-weight restoration, and auto-detects CUDA.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train (typically :class:`ExpressionMLP`).
    task : {'classification', 'regression'}
        Determines the loss function and evaluation metric:
        - ``'classification'`` -> CrossEntropyLoss, accuracy reported.
        - ``'regression'``     -> MSELoss, R^2 reported.
        Default: ``'classification'``.
    device : str or torch.device, optional
        Compute device (``'cpu'``, ``'cuda'``, ``'mps'``, ...).
        If ``None``, CUDA is used when available, otherwise CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        task: str = "classification",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if task not in ("classification", "regression"):
            raise ValueError(
                f"task must be 'classification' or 'regression', got '{task}'."
            )

        # Auto-detect compute device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.task = task

        # Move model to the selected device
        self.model = model.to(self.device)
        logger.info(
            f"ExpressionTrainer initialised — task='{task}', "
            f"device='{self.device}'."
        )

        # Loss function varies by task
        if task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        # Stores the best model weights found during fit() for early-stopping
        # restoration
        self._best_weights: Optional[dict] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_loader(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """Wrap raw arrays in an ExpressionDataset and return a DataLoader."""
        return DataLoader(
            ExpressionDataset(X, y),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def _run_epoch(
        self,
        loader: DataLoader,
        optimiser: Optional[torch.optim.Optimizer],
    ):
        """
        Run one complete pass over a DataLoader.

        Parameters
        ----------
        loader : DataLoader
        optimiser : Optimizer or None
            Provide an optimiser for a *training* pass (gradients computed,
            weights updated).  Pass ``None`` for *validation / evaluation*
            (model in eval mode, no gradient computation).

        Returns
        -------
        avg_loss : float
        avg_metric : float
            Accuracy (classification) or R^2 (regression) for this epoch.
        """
        training = optimiser is not None
        self.model.train(training)

        total_loss = 0.0
        all_preds: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        # Use no_grad only for validation / evaluation passes
        grad_ctx = torch.enable_grad() if training else torch.no_grad()
        with grad_ctx:
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch)

                if self.task == "classification":
                    # CrossEntropyLoss expects Long targets and (N, C) logits
                    loss = self.criterion(logits, y_batch.long())
                    preds = logits.argmax(dim=1).float()
                else:
                    # MSELoss: squeeze model output from (N,1) to (N,)
                    loss = self.criterion(logits.squeeze(1), y_batch)
                    preds = logits.squeeze(1)

                if training:
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                total_loss += loss.item() * X_batch.size(0)
                all_preds.append(preds.detach().cpu())
                all_targets.append(y_batch.detach().cpu())

        n_samples = sum(t.shape[0] for t in all_targets)
        avg_loss = total_loss / n_samples

        preds_cat = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)

        if self.task == "classification":
            avg_metric = (preds_cat == targets_cat.float()).float().mean().item()
        else:
            # R^2 = 1 - SS_res / SS_tot
            ss_res = ((targets_cat - preds_cat) ** 2).sum().item()
            ss_tot = ((targets_cat - targets_cat.mean()) ** 2).sum().item()
            avg_metric = 1.0 - ss_res / (ss_tot + 1e-8)

        return avg_loss, avg_metric

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series],
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        patience: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Train the model with early stopping and best-weight restoration.

        If validation loss does not improve by more than 1e-6 for ``patience``
        consecutive epochs, training halts and the best weights are restored.

        Parameters
        ----------
        X_train, y_train : array-like
            Training features and labels.
        X_val, y_val : array-like
            Validation features and labels.
        epochs : int
            Maximum number of training epochs. Default: 100.
        lr : float
            Adam learning rate. Default: 1e-3.
        batch_size : int
            Mini-batch size. Default: 32.
        patience : int
            Early stopping patience. Default: 10.

        Returns
        -------
        dict
            Training history with keys
            ``{'train_loss', 'val_loss', 'train_acc', 'val_acc'}``,
            each a list of per-epoch float values.
        """
        train_loader = self._make_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader   = self._make_loader(X_val,   y_val,   batch_size, shuffle=False)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }

        best_val_loss = float("inf")
        epochs_no_improve = 0
        self._best_weights = copy.deepcopy(self.model.state_dict())

        logger.info(
            f"Starting training: max_epochs={epochs}, lr={lr}, "
            f"batch_size={batch_size}, patience={patience}."
        )

        for epoch in range(1, epochs + 1):
            train_loss, train_metric = self._run_epoch(train_loader, optimiser)
            val_loss,   val_metric   = self._run_epoch(val_loader,   None)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_metric)
            history["val_acc"].append(val_metric)

            metric_label = "acc" if self.task == "classification" else "R2"
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f}  train_{metric_label}={train_metric:.4f} | "
                f"val_loss={val_loss:.4f}    val_{metric_label}={val_metric:.4f}"
            )

            # --- Early stopping ---
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self._best_weights = copy.deepcopy(self.model.state_dict())
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no val improvement for {patience} epochs)."
                    )
                    break

        # Restore the best weights seen during training
        self.model.load_state_dict(self._best_weights)
        logger.info("Best weights restored after early stopping.")
        return history

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Generate predictions for input expression data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_genes)

        Returns
        -------
        np.ndarray
            - Classification: integer class indices, shape (n_samples,).
            - Regression: predicted values, shape (n_samples,).
        """
        self.model.eval()
        X_tensor = torch.from_numpy(_to_numpy(X)).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)

        if self.task == "classification":
            return logits.argmax(dim=1).cpu().numpy()
        else:
            return logits.squeeze(1).cpu().numpy()

    def evaluate(
        self,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
    ) -> Dict[str, float]:
        """
        Evaluate the model on a held-out test set.

        Parameters
        ----------
        X_test, y_test : array-like

        Returns
        -------
        dict
            - Classification: ``{'loss': float, 'accuracy': float}``.
            - Regression:     ``{'loss': float, 'r2': float}``.
        """
        loader = self._make_loader(X_test, y_test, batch_size=64, shuffle=False)
        loss, metric = self._run_epoch(loader, optimiser=None)

        if self.task == "classification":
            result = {"loss": loss, "accuracy": metric}
        else:
            result = {"loss": loss, "r2": metric}

        logger.info(f"Test evaluation: {result}")
        return result

    def save_checkpoint(self, path: str) -> None:
        """
        Save the model's state dict to a ``.pt`` file.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``'results/mlp_model.pt'``).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model checkpoint saved to '{path}'.")

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
    ) -> matplotlib.figure.Figure:
        """
        Plot loss and accuracy / R^2 curves for training and validation.

        Parameters
        ----------
        history : dict
            The dict returned by :meth:`fit`, with keys
            ``'train_loss'``, ``'val_loss'``, ``'train_acc'``, ``'val_acc'``.

        Returns
        -------
        matplotlib.figure.Figure
            Two-panel figure:
            - Left:  loss curves (train vs. validation).
            - Right: accuracy or R^2 curves (train vs. validation).
        """
        epochs_range = range(1, len(history["train_loss"]) + 1)
        metric_label = "Accuracy" if self.task == "classification" else "R\u00b2"

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss panel
        ax = axes[0]
        ax.plot(epochs_range, history["train_loss"],
                label="Train", color="#2196F3", linewidth=1.5)
        ax.plot(epochs_range, history["val_loss"],
                label="Validation", color="#F44336", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        # Metric panel
        ax = axes[1]
        ax.plot(epochs_range, history["train_acc"],
                label=f"Train {metric_label}", color="#2196F3", linewidth=1.5)
        ax.plot(epochs_range, history["val_acc"],
                label=f"Val {metric_label}", color="#F44336", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.suptitle("ExpressionMLP — Training Curves", fontsize=13, y=1.01)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# 4. Model — Variational Autoencoder (VAE)
# ---------------------------------------------------------------------------

class ExpressionVAE(nn.Module):
    """
    Variational Autoencoder for non-linear dimensionality reduction of
    high-dimensional gene expression matrices.

    Architecture
    ------------
    Encoder : input_dim -> hidden_dims -> (mu, logvar) of size latent_dim
    Decoder : latent_dim -> reversed hidden_dims -> input_dim

    The reparameterisation trick is used during training to allow gradients
    to flow through the stochastic sampling step.  During evaluation the
    encoder returns ``mu`` directly, making ``transform()`` deterministic
    and fully reproducible.

    Parameters
    ----------
    input_dim : int
        Number of input features (genes after filtering).
    hidden_dims : list of int
        Encoder hidden layer sizes in order, e.g. ``[256, 128]``.
        The decoder uses the reverse order automatically.
    latent_dim : int
        Dimensionality of the latent space (bottleneck). Default: 32.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int = 32,
    ) -> None:
        super().__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size.")

        # --- Encoder ---
        # Compresses input_dim down through hidden_dims to a bottleneck,
        # then projects to mu and logvar separately.
        encoder_layers: List[nn.Module] = []
        in_features = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_features, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
            ])
            in_features = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Separate projection heads for the mean and log-variance
        self.fc_mu     = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)

        # --- Decoder ---
        # Mirrors the encoder in reverse order, reconstructing back to input_dim.
        decoder_layers: List[nn.Module] = []
        in_features = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_features, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
            ])
            in_features = h_dim

        # Final reconstruction layer — no activation (expression values are
        # log-transformed and can be any real number)
        decoder_layers.append(nn.Linear(in_features, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor):
        """
        Pass input through the encoder and return mu and logvar.

        Parameters
        ----------
        x : Tensor, shape (batch, input_dim)

        Returns
        -------
        mu, logvar : Tensor, each shape (batch, latent_dim)
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterisation trick: z = mu + eps * std.

        During **training** (``self.training=True``) a random epsilon is
        sampled so gradients can flow.  During **evaluation** ``mu`` is
        returned directly, making the encoding deterministic and reproducible.

        Parameters
        ----------
        mu : Tensor, shape (batch, latent_dim)
        logvar : Tensor, shape (batch, latent_dim)

        Returns
        -------
        z : Tensor, shape (batch, latent_dim)
        """
        if self.training:
            # std = exp(0.5 * logvar); eps ~ N(0, I)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        # Eval mode: return the deterministic mean
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent vector back to input space.

        Parameters
        ----------
        z : Tensor, shape (batch, latent_dim)

        Returns
        -------
        recon_x : Tensor, shape (batch, input_dim)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        Full VAE forward pass: encode -> reparameterise -> decode.

        Parameters
        ----------
        x : Tensor, shape (batch, input_dim)

        Returns
        -------
        recon_x : Tensor, shape (batch, input_dim)
        mu      : Tensor, shape (batch, latent_dim)
        logvar  : Tensor, shape (batch, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# ---------------------------------------------------------------------------
# 5. Loss — VAE combined loss with beta-VAE KL annealing support
# ---------------------------------------------------------------------------

class VAELoss(nn.Module):
    """
    Combined VAE loss: reconstruction loss (MSE) + beta * KL divergence.

    The KL divergence term measures how far the learned posterior
    q(z|x) = N(mu, sigma^2) deviates from the standard normal prior
    p(z) = N(0, I):

        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    The ``beta`` coefficient implements beta-VAE annealing: starting beta
    at 0 and ramping it up to 1 during early training prevents posterior
    collapse, where the decoder learns to ignore the latent space entirely.

    Parameters
    ----------
    reduction : {'mean', 'sum'}
        Reduction applied to the MSE reconstruction loss.
        Default: ``'mean'``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.recon_loss_fn = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute the total VAE loss.

        Parameters
        ----------
        recon_x : Tensor, shape (batch, input_dim)
            Decoder reconstruction output.
        x : Tensor, shape (batch, input_dim)
            Original input (target for reconstruction).
        mu : Tensor, shape (batch, latent_dim)
            Encoder mean.
        logvar : Tensor, shape (batch, latent_dim)
            Encoder log-variance.
        beta : float
            Weight applied to the KL divergence term. Use values in [0, 1]
            for beta-VAE annealing (0 = pure reconstruction, 1 = standard VAE).
            Default: 1.0.

        Returns
        -------
        total_loss : Tensor (scalar)
        """
        # Reconstruction loss: how well the decoder reproduces the input
        recon_loss = self.recon_loss_fn(recon_x, x)

        # KL divergence: closed-form for Gaussian posterior vs N(0,I) prior.
        # Mean over the batch for consistency with MSE 'mean' reduction.
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )

        return recon_loss + beta * kl_loss


# ---------------------------------------------------------------------------
# 6. Trainer — VAE (unsupervised; separate from ExpressionTrainer)
# ---------------------------------------------------------------------------

class VAETrainer:
    """
    Unsupervised training loop for :class:`ExpressionVAE`.

    Implements KL-annealing (beta warmup) to avoid posterior collapse and
    provides a deterministic ``transform()`` method for latent-space
    projection and downstream visualisation.

    Parameters
    ----------
    model : ExpressionVAE
        The VAE model to train.
    device : str or torch.device, optional
        Compute device.  Auto-detects CUDA if ``None``.
    """

    def __init__(
        self,
        model: "ExpressionVAE",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = VAELoss(reduction="mean")

        logger.info(
            f"VAETrainer initialised — latent_dim={model.latent_dim}, "
            f"device='{self.device}'."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_loader(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """
        Wrap a feature matrix in a DataLoader for the VAE.

        The VAE is unsupervised so we pass the same array as both X and y
        (y is not used during training, but ExpressionDataset requires it).
        """
        X_arr = _to_numpy(X)
        # Dummy targets — not used in the VAE loss
        dummy_y = np.zeros(X_arr.shape[0], dtype=np.float32)
        return DataLoader(
            ExpressionDataset(X_arr, dummy_y),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    @staticmethod
    def _compute_beta(
        epoch: int,
        beta_start: float,
        beta_end: float,
        beta_warmup_epochs: int,
    ) -> float:
        """
        Compute the annealed beta value for a given epoch.

        Linearly ramps from ``beta_start`` to ``beta_end`` over
        ``beta_warmup_epochs`` epochs, then holds at ``beta_end``.
        """
        if beta_warmup_epochs <= 0:
            return beta_end
        progress = min(epoch / beta_warmup_epochs, 1.0)
        return beta_start + progress * (beta_end - beta_start)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_vae(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        beta_warmup_epochs: int = 20,
    ) -> Dict[str, List[float]]:
        """
        Train the VAE with KL-annealing (beta warmup).

        KL annealing schedule:
            epoch 1 → beta = beta_start
            epoch beta_warmup_epochs → beta = beta_end
            epoch > beta_warmup_epochs → beta = beta_end (constant)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_genes)
            log2(CPM+1) expression matrix.
        epochs : int
            Maximum training epochs. Default: 100.
        lr : float
            Adam learning rate. Default: 1e-3.
        batch_size : int
            Mini-batch size. Default: 32.
        beta_start : float
            Initial KL weight (0 = pure reconstruction). Default: 0.0.
        beta_end : float
            Final KL weight (1 = standard VAE). Default: 1.0.
        beta_warmup_epochs : int
            Number of epochs to linearly ramp beta. Default: 20.

        Returns
        -------
        dict
            Training history with keys
            ``{'train_loss', 'recon_loss', 'kl_loss'}``,
            each a list of per-epoch floats.
        """
        loader = self._make_loader(X, batch_size=batch_size, shuffle=True)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "recon_loss": [],
            "kl_loss":    [],
        }

        logger.info(
            f"Starting VAE training: epochs={epochs}, lr={lr}, "
            f"batch_size={batch_size}, beta_warmup={beta_warmup_epochs}."
        )

        for epoch in range(1, epochs + 1):
            self.model.train()
            beta = self._compute_beta(epoch, beta_start, beta_end, beta_warmup_epochs)

            epoch_total   = 0.0
            epoch_recon   = 0.0
            epoch_kl      = 0.0
            n_samples     = 0

            for X_batch, _ in loader:   # _ = dummy targets, not used
                X_batch = X_batch.to(self.device)
                batch_n = X_batch.size(0)

                recon_x, mu, logvar = self.model(X_batch)

                # Reconstruction component (unweighted)
                recon_loss = nn.functional.mse_loss(recon_x, X_batch, reduction="mean")
                # KL component (unweighted)
                kl_loss = -0.5 * torch.mean(
                    torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                )
                total_loss = recon_loss + beta * kl_loss

                optimiser.zero_grad()
                total_loss.backward()
                optimiser.step()

                epoch_total  += total_loss.item() * batch_n
                epoch_recon  += recon_loss.item() * batch_n
                epoch_kl     += kl_loss.item()    * batch_n
                n_samples    += batch_n

            avg_total = epoch_total  / n_samples
            avg_recon = epoch_recon  / n_samples
            avg_kl    = epoch_kl     / n_samples

            history["train_loss"].append(avg_total)
            history["recon_loss"].append(avg_recon)
            history["kl_loss"].append(avg_kl)

            logger.info(
                f"VAE Epoch {epoch:3d}/{epochs} | "
                f"beta={beta:.3f} | "
                f"total={avg_total:.4f}  recon={avg_recon:.4f}  kl={avg_kl:.4f}"
            )

        logger.info("VAE training complete.")
        return history

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Encode an expression matrix into latent-space ``mu`` coordinates.

        The model is set to eval mode so that ``reparameterize()`` returns
        ``mu`` directly — making this method fully deterministic.  Calling
        ``transform()`` twice on the same input will always produce identical
        results.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_genes)

        Returns
        -------
        np.ndarray, shape (n_samples, latent_dim)
            Deterministic latent mean vectors (``mu``).
        """
        self.model.eval()
        X_tensor = torch.from_numpy(_to_numpy(X)).to(self.device)

        with torch.no_grad():
            mu, _ = self.model.encode(X_tensor)
            # reparameterize in eval mode also returns mu, but calling encode
            # directly is cleaner and makes the intent explicit.

        return mu.cpu().numpy()

    def save_checkpoint(self, path: str) -> None:
        """
        Save the VAE model's state dict to a ``.pt`` file.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``'results/vae_model.pt'``).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"VAE checkpoint saved to '{path}'.")

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
    ) -> matplotlib.figure.Figure:
        """
        Plot total loss, reconstruction loss, and KL loss per epoch.

        Parameters
        ----------
        history : dict
            The dict returned by :meth:`fit_vae`, with keys
            ``'train_loss'``, ``'recon_loss'``, ``'kl_loss'``.

        Returns
        -------
        matplotlib.figure.Figure
            Three-panel figure — one panel per loss component.
        """
        epochs_range = range(1, len(history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        panels = [
            ("train_loss", "Total Loss",           "#424242"),
            ("recon_loss", "Reconstruction (MSE)", "#1976D2"),
            ("kl_loss",    "KL Divergence",        "#E64A19"),
        ]

        for ax, (key, title, colour) in zip(axes, panels):
            ax.plot(epochs_range, history[key], color=colour, linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.5)

        fig.suptitle("ExpressionVAE — Training Curves", fontsize=13, y=1.01)
        fig.tight_layout()
        return fig
