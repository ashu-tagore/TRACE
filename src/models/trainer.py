import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import yaml

from src.models.lstm_vae import LSTMVAE


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_vae_loss(x: torch.Tensor, reconstruction: torch.Tensor,
                     mu: torch.Tensor, log_var: torch.Tensor,
                     beta: float) -> tuple[torch.Tensor, float, float]:
    """Returns (total_loss_tensor, recon_loss_scalar, kl_loss_scalar).

    Both terms are summed over event dimensions (timesteps + features for
    reconstruction; latent dims for KL) and averaged over the batch. This
    is the proper ELBO formulation. Using `reduction='mean'` for both
    would silently rescale the reconstruction term by the number of
    elements (30 * 63 = 1890), making it ~1890x smaller per item than
    the KL term and giving the KL term enormous relative weight once
    beta reaches 1.0 — the classic recipe for posterior collapse.

    KL divergence between N(mu, sigma^2) and N(0,1) has a closed form:
    -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2). No sampling needed.
    """
    batch_size = x.shape[0]
    recon_loss = nn.functional.mse_loss(reconstruction, x, reduction="sum") / batch_size
    kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss.item(), kl_loss.item()


def beta_for_epoch(epoch: int, beta_max: float, anneal_epochs: int) -> float:
    """Returns beta linearly increasing from 0 to beta_max over anneal_epochs."""
    if epoch >= anneal_epochs:
        return beta_max
    return beta_max * (epoch / anneal_epochs)


def run_one_epoch(model: LSTMVAE, loader: DataLoader,
                  optimizer: torch.optim.Optimizer,
                  beta: float, is_training: bool,
                  device: torch.device) -> tuple[float, float, float]:
    """Returns (avg_total, avg_recon, avg_kl) over the entire loader."""
    model.train(is_training)

    total_losses, recon_losses, kl_losses = [], [], []
    grad_context = torch.enable_grad() if is_training else torch.no_grad()

    with grad_context:
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction, mu, log_var = model(batch)
            loss, recon, kl = compute_vae_loss(batch, reconstruction, mu, log_var, beta)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_losses.append(loss.item())
            recon_losses.append(recon)
            kl_losses.append(kl)

    return np.mean(total_losses), np.mean(recon_losses), np.mean(kl_losses)


def train():
    config = load_config()
    mc     = config["model"]
    tc     = config["training_loop"]
    mfc    = config["mlflow"]

    train_seqs = np.load("data/processed/train_sequences.npy")
    val_seqs   = np.load("data/processed/val_sequences.npy")

    train_loader = DataLoader(TensorDataset(torch.tensor(train_seqs)),
                              batch_size=tc["batch_size"], shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(val_seqs)),
                              batch_size=tc["batch_size"])

    n_features = train_seqs.shape[2]
    seq_len    = train_seqs.shape[1]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = LSTMVAE(
        input_size=n_features,
        hidden_size=mc["hidden_size"],
        num_layers=mc["num_layers"],
        latent_dim=mc["latent_dim"],
        seq_len=seq_len,
        dropout=mc["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])

    mlflow.set_experiment(mfc["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params({**mc, **tc, "n_features": n_features, "seq_len": seq_len})

        best_val_recon             = float("inf")
        epochs_without_improvement = 0

        for epoch in range(tc["epochs"]):
            beta = beta_for_epoch(epoch, mc["beta_max"], mc["beta_anneal_epochs"])

            train_total, train_recon, train_kl = run_one_epoch(
                model, train_loader, optimizer, beta, is_training=True, device=device
            )
            val_total, val_recon, val_kl = run_one_epoch(
                model, val_loader, optimizer, beta, is_training=False, device=device
            )

            mlflow.log_metrics({
                "train/total": train_total, "train/recon": train_recon,
                "train/kl":    train_kl,    "val/total":   val_total,
                "val/recon":   val_recon,   "val/kl":      val_kl,
                "beta":        beta,
            }, step=epoch)

            # Checkpoint on val_recon, not val_total — total loss is a
            # moving target during beta annealing (the formula literally
            # changes each epoch), so the "best" epoch by total loss is
            # almost always the early epoch where beta is near zero.
            # val_recon has stable meaning across the whole run.
            val_has_improved = val_recon < best_val_recon
            if val_has_improved:
                best_val_recon             = val_recon
                epochs_without_improvement = 0
                mlflow.pytorch.log_model(model, artifact_path="lstm_vae")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= tc["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

            print(f"Epoch {epoch:03d} | beta={beta:.2f} | "
                  f"train={train_total:.4f} | val={val_total:.4f}")

        run_id    = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/lstm_vae"
        mlflow.register_model(model_uri, name=mfc["model_name"])
        print(f"Registered model: {mfc['model_name']}")


if __name__ == "__main__":
    train()