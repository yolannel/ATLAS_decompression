import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from datagen import get_dataloader, create_file_pairs
from plotting import plot_log_residual_contour, plot_histogram
from models import Autoencoder1D, ConvAutoencoder1D
import os
from pathlib import Path
from tqdm import tqdm

os.environ["OPENBLAS_NUM_THREADS"] = "1"

def compute_log_residuals(x_true, x_recon):
    residual = x_true - x_recon
    log_x_true = np.log10(np.abs(x_true) + 1e-12)
    log_residual = np.sign(residual) * np.log10(np.abs(residual) + 1e-12)
    return log_x_true, log_residual, residual

def compute_rule_based_residual_magnitude(log_x_true, m=23, epsilon=0.3):
    delta_x = np.log10(2)
    C = -m * delta_x + epsilon
    x_step = np.floor(log_x_true / delta_x) * delta_x
    y_upper = x_step + C
    y_lower = -x_step - C
    y_mid = (y_upper + y_lower) / 2
    return y_mid, y_upper, y_lower, x_step

def plot_rule_based_bounds(log_x_true, log_residual, y_upper, y_lower, x_step, m, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(log_x_true, log_residual, s=5, alpha=0.2, label="Residuals")
    sort_idx = np.argsort(x_step)
    plt.plot(x_step[sort_idx], y_upper[sort_idx], 'r--', label=f"$y = x - {m}\\log_{{10}}2 + \\epsilon$")
    plt.plot(x_step[sort_idx], y_lower[sort_idx], 'b--', label=f"$y = -x + {m}\\log_{{10}}2 - \\epsilon$")
    plt.xlabel("log10(|x_true|)")
    plt.ylabel("Signed log10 residual")
    plt.title("Rule-Based Residual Bounds")
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(Path(save_path) / "rule_based_bounds.png")
        print(f"Rule-based bounds plot saved to {Path(save_path) / 'rule_based_bounds.png'}")
        plt.close()
    else:
        plt.show()

class ResidualSignModel:
    def __init__(self, model, device='cpu', hidden_dim=32, noise_std=0.01, save_path="./tmp/"):
        self.device = device
        self.noise_std = noise_std
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.save_path = Path(save_path)
        self.mean_src = None
        self.std_src = None
        os.makedirs(self.save_path, exist_ok=True)

    def fit(self, dataloader, epochs=100, batch_size=128):
        """
        Train the autoencoder using synthetic noisy batches from single-event dataloader.

        Args:
            dataloader: yields (x_compressed, residual) where x_compressed is shape (1000, 1)
            epochs: number of training epochs
            batch_size: number of noisy variants per sample
        """
        self.model.train()
        loss_history = []
        plotted_og = False

        total_sum = 0.0
        total_sq_sum = 0.0
        total_count = 0

        for x_compressed, residual in dataloader:
            x_clean = x_compressed.squeeze(1)
            total_sum += x_clean.sum().item()
            total_sq_sum += (x_clean ** 2).sum().item()
            total_count += x_clean.numel()

        global_mean = total_sum / total_count
        global_std = (total_sq_sum / total_count - global_mean ** 2) ** 0.5

        self.mean_src = global_mean
        self.std_src = global_std

        for epoch in range(epochs):
            running_loss = 0.0
            total_samples = 0

            for (x_compressed, residual) in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                x_compressed = x_compressed.to(self.device)  # shape: (1000, 1)
                x_clean = x_compressed.squeeze(1)            # shape: (1000,)
                residual = residual.to(self.device)
                # Expand to (batch_size, 1000), then add channel dimension → (batch_size, 1, 1000)
                x_clean = (x_clean - self.mean_src) / (self.std_src + 1e-8)
                x_clean_batch = x_clean.unsqueeze(0).expand(batch_size, -1)     # (batch_size, 1000)
                x_clean_batch = x_clean_batch.unsqueeze(1)                      # (batch_size, 1, 1000)

                noise = torch.randn_like(x_clean_batch) * self.noise_std
                x_noisy_batch = x_clean_batch + noise                           # (batch_size, 1, 1000)

                # Optional: visualize on one sample
                if not plotted_og:
                    x_recon = (x_clean + residual).squeeze().cpu().numpy()
                    plot_log_residual_contour(
                        x_clean.squeeze().cpu().numpy(),
                        x_recon,
                        gmm=None,
                        varname="pt",
                        save_path=self.save_path
                    )
                    # print(f"clean batch first sample type: {type(x_clean_batch[0])}, noisy batch first sample type: {type(x_noisy_batch[0])}")
                    plot_histogram((x_clean_batch[0].squeeze().cpu().numpy(), x_noisy_batch[0].squeeze().cpu().numpy()), varnames=["clean","noisy"], save_path=self.save_path)
                    plotted_og = True
                pred = self.model(x_noisy_batch)                  # shape: (batch_size, 1, 1000)
                loss = self.criterion(pred, x_clean_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * batch_size
                total_samples += batch_size

                last_noisy = x_noisy_batch[-1].squeeze().detach().cpu().numpy()
                last_denoised = pred[-1].squeeze().detach().cpu().numpy()


            avg_loss = running_loss / total_samples
            loss_history.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        # Save the model after training if save_path is provided
        if self.save_path is not None:
            torch.save(self.model.state_dict(), self.save_path / "residual_sign_model.pth")
            print(f"Model saved to {self.save_path}")
            # Plot and save loss history
            plt.figure()
            plt.plot(loss_history, marker='o')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss History")
            plt.grid(True)
            plt.savefig(self.save_path / "loss_history.png")
            plt.close()
            print(f"Loss history plot saved to {self.save_path / 'loss_history.png'}")
            
            # Plot last noisy and denoised input/output as histograms
            plt.figure()
            last_noisy = (last_noisy * self.std_src + self.mean_src)
            last_denoised = (last_denoised * self.std_src + self.mean_src)
            plt.hist(last_noisy, bins=50, alpha=0.5, label="Noisy Input")
            plt.hist(last_denoised, bins=50, alpha=0.5, label="Denoised Output")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.title("Last Noisy vs Denoised Histogram")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.save_path / "last_noisy_vs_denoised_hist.png")
            plt.close()
            print(f"Last noisy/denoised histogram saved to {self.save_path / 'last_noisy_vs_denoised_hist.png'}")


    def predict(self, x_noisy):
        """
        Reconstruct clean data from noisy input.
        
        Args:
            x_noisy: np.array shape (N,)
            
        Returns:
            np.array shape (N,) reconstructed (denoised) data
        """
        self.model.eval()
        x_noisy = (x_noisy - self.mean_src) / (self.std_src + 1e-8)
        x_tensor = torch.tensor(x_noisy, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            recon = self.model(x_tensor).cpu().numpy().flatten()
        return recon

def hybrid_residual_corrector(x_compressed, x_recon, m=23, epsilon=0.3, threshold=1, mean_src=None, std_src=None, mask_range_factor=2.0):
    """
    Hybrid residual corrector with sign-aware correction logic.

    Args:
        x_compressed: compressed values (array)
        x_recon: reconstructed/compressed values (array)
        m, epsilon, threshold: rule-based parameters
        mean_src, std_src: normalization stats
        mask_range_factor: mask out corrected values if abs(x_recon_corrected - x_recon) > mask_range_factor * std(x_recon)
    Returns:
        dict with correction results
    """
    # De-normalize if stats are provided
    # if mean_src is not None and std_src is not None:
    #     x_compressed = x_compressed * std_src + mean_src
    #     x_recon = x_recon * std_src + mean_src
    # else:
    #     print("No normalization stats provided, using raw input.")

    log_x_true, log_residual, residual = compute_log_residuals(x_compressed, x_recon)
    y_mid, y_upper, y_lower, x_step = compute_rule_based_residual_magnitude(log_x_true, m, epsilon)

    corrected_residual = np.zeros_like(log_residual)
    for i in range(len(log_residual)):
        res = log_residual[i]
        # If residual is positive, consider negative correction
        if res > 0:
            neg_res = -res
            if (neg_res - y_lower[i] < threshold) and (neg_res > y_lower[i]):
                corrected_residual[i] = neg_res
                print(f"[INFO] Index {i}: Positive residual {res:.4f}, applying negative correction {neg_res:.4f} (y_lower={y_lower[i]:.4f}, threshold={threshold})")
            else:
                corrected_residual[i] = 0.0
        # If residual is negative, consider positive correction
        elif res < 0:
            pos_res = -res
            if (y_upper[i] - pos_res < threshold) and (pos_res < y_upper[i]):
                corrected_residual[i] = pos_res
                print(f"[INFO] Index {i}: Negative residual {res:.4f}, applying positive correction {pos_res:.4f} (y_upper={y_upper[i]:.4f}, threshold={threshold})")
            else:
                corrected_residual[i] = 0.0
        else:
            corrected_residual[i] = 0.0

    x_recon_corrected_log = log_x_true + corrected_residual
    # Invert log transform to get corrected values
    x_recon_corrected = np.sign(x_recon_corrected_log) * 10 ** np.abs(x_recon_corrected_log)

    # Mask out values that are too far from compressed
    std_recon = np.std(x_recon)
    mask = np.abs(x_recon_corrected - x_recon) < mask_range_factor * std_recon
    x_recon_corrected_masked = np.where(mask, x_recon_corrected, x_recon)

    # Optionally, update log values for masked output
    x_recon_corrected_log_masked = np.log10(np.abs(x_recon_corrected_masked) + 1e-12)
    
    return {
        "log_x_true": log_x_true,
        "log_residual": log_residual,
        "x_step": x_step,
        "y_upper": y_upper,
        "y_lower": y_lower,
        "residual_pred": corrected_residual,
        "x_recon_corrected_log": x_recon_corrected_log_masked,
        "x_recon_corrected": x_recon_corrected_masked,
        "mask": mask,
    }

def main():
    branch = "AnalysisElectronsAuxDyn"
    varnames = ["pt"]

    batch_size = 1000
    dataloader = get_dataloader(branch=branch, varnames=varnames, batch_size=batch_size, shuffle=False)

    module = ConvAutoencoder1D()
    model = ResidualSignModel(module, noise_std=0.1, device='cuda' if torch.cuda.is_available() else 'cpu', save_path = "./tmp")

    # Grab one batch from dataloader
    for x_compressed, residual_tensor in dataloader:
        x_compressed = x_compressed[:, 0].numpy()
        x_recon = x_compressed + residual_tensor[:, 0].numpy()
        x_data_true = x_recon.copy()
        plot_log_residual_contour(x_compressed, x_recon, gmm=None, varname="pt", save_path=Path(model.save_path) / f"log_residual_contour_pt.png")
        break

    load_model = True

    model_path = model.save_path / "residual_sign_model.pth"
    stats_path = model.save_path / "norm_stats.npz"

    if load_model and model_path.exists() and stats_path.exists():
        print(f"Loading model from {model_path}")
        model.model.load_state_dict(torch.load(model_path, map_location=model.device))
        stats = np.load(stats_path)
        model.mean_src = stats["mean_src"].item()
        model.std_src = stats["std_src"].item()
        print("Model and normalization stats loaded. Skipping fit().")
    else:
        model.fit(dataloader, epochs=30)
        np.savez(stats_path, mean_src=model.mean_src, std_src=model.std_src)

    # --- Evaluate hybrid_residual_corrector performance ---
    result = hybrid_residual_corrector(x_compressed, x_recon, m=10, mean_src=model.mean_src, std_src=model.std_src)
    plot_rule_based_bounds(
        result["log_x_true"],
        result["log_residual"],
        result["y_upper"],
        result["y_lower"],
        result["x_step"],
        m=10,
        save_path=model.save_path
    )

    x_recon_corrected = np.sign(result["x_recon_corrected_log"]) * 10 ** np.abs(result["x_recon_corrected_log"])
    x_true = x_compressed + residual_tensor[:, 0].numpy()

    plot_log_residual_contour(
        x_true,
        x_recon_corrected,
        gmm=None,
        varname="pt",
        save_path=Path(model.save_path) / f"hybrid_corrected_log_residual_contour_pt.png"
    )

    mse = np.mean((x_true - x_recon_corrected) ** 2)
    mae = np.mean(np.abs(x_true - x_recon_corrected))
    print(f"Hybrid Residual Corrector MSE: {mse:.6f}, MAE: {mae:.6f}")

    plt.figure()
    plt.hist(x_true - x_recon_corrected, bins=50, alpha=0.7)
    plt.xlabel("Residual (True - Corrected)")
    plt.ylabel("Frequency")
    plt.title("Residuals after Hybrid Correction")
    plt.grid(True)
    plt.savefig(model.save_path / "hybrid_correction_residuals.png")
    plt.close()
    print(f"Hybrid correction residuals histogram saved to {model.save_path / 'hybrid_correction_residuals.png'}")

    # --- New Residual Analysis: Model Output vs True ---
    # Apply same normalization as in fit before predict
    x_compressed = torch.tensor(x_compressed).unsqueeze(0).to(model.device)
    x_noisy = ((x_compressed - model.mean_src) / (model.std_src + 1e-8))
    x_model_pred = model.predict(x_noisy).squeeze()
    # x_model_pred = x_model_pred * model.std_src + model.mean_src
    x_compressed = x_compressed.squeeze()
    model_residual = x_true - x_model_pred.squeeze()

    plt.figure()
    plt.hist(model_residual, bins=50, alpha=0.7)
    plt.xlabel("Residual (True - Model Output)")
    plt.ylabel("Frequency")
    plt.title("Residuals: Model Output vs True")
    plt.grid(True)
    plt.savefig(model.save_path / "model_vs_true_residuals.png")
    plt.close()
    print(f"Model vs true residuals histogram saved to {model.save_path / 'model_vs_true_residuals.png'}")

    # --- Plot histograms of compressed, predicted, and ground truth ---
    plt.figure()
    plt.hist(x_compressed.detach().cpu().numpy(), bins=50, alpha=0.5, label="Compressed")
    plt.hist(x_model_pred.squeeze() + x_compressed.detach().cpu().numpy(), bins=50, alpha=0.5, label="Predicted (Model Output + Compressed)")
    plt.hist(x_true, bins=50, alpha=0.5, label="Ground Truth (Compressed + Residual)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Comparison: Compressed, Predicted, Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.savefig(model.save_path / "compressed_predicted_groundtruth_hist.png")
    plt.close()
    print(f"Comparison histogram saved to {model.save_path / 'compressed_predicted_groundtruth_hist.png'}")

if __name__ == "__main__":
    main()
