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

def plot_rule_based_bounds(log_x_true, log_residual, y_upper, y_lower, x_step, m):
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
    plt.show()

class ResidualSignModel:
    def __init__(self, model, device='cpu', hidden_dim=32, noise_std=0.01):
        self.device = device
        self.noise_std = noise_std
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def fit(self, dataloader, epochs=100, batch_size=128):
        """
        Train the autoencoder using synthetic noisy batches from single-event dataloader.

        Args:
            dataloader: yields (x_compressed, residual) where x_compressed is shape (1000, 1)
            epochs: number of training epochs
            batch_size: number of noisy variants per sample
        """
        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            total_samples = 0

            for (x_compressed, residual) in dataloader:
                x_compressed = x_compressed.to(self.device)  # shape: (1000, 1)
                x_clean = x_compressed.squeeze(1)            # shape: (1000,)

                # Expand to (batch_size, 1000), then add channel dimension → (batch_size, 1, 1000)
                x_clean_batch = x_clean.unsqueeze(0).expand(batch_size, -1)     # (batch_size, 1000)
                x_clean_batch = x_clean_batch.unsqueeze(1)                      # (batch_size, 1, 1000)

                noise = torch.randn_like(x_clean_batch) * self.noise_std
                x_noisy_batch = x_clean_batch + noise                           # (batch_size, 1, 1000)

                # Optional: visualize on one sample
                if epoch == 0:
                    x_recon = (x_clean + residual).squeeze().cpu().numpy()
                    plot_log_residual_contour(
                        x_clean.squeeze().cpu().numpy(),
                        x_recon,
                        gmm=None,
                        varname="pt"
                    )
                    print(f"x_clean shape: {x_clean_batch.shape}, x_noisy_batch shape: {x_noisy_batch.shape}")
                    # print(f"clean batch first sample type: {type(x_clean_batch[0])}, noisy batch first sample type: {type(x_noisy_batch[0])}")
                    plot_histogram((x_clean_batch[0].squeeze().cpu().numpy(), x_noisy_batch[0].squeeze().cpu().numpy()), varnames=["clean","noisy"])
                    exit()

                pred = self.model(x_noisy_batch)                  # shape: (batch_size, 1, 1000)
                loss = self.criterion(pred, x_clean_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * batch_size
                total_samples += batch_size

            avg_loss = running_loss / total_samples
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")


    def predict(self, x_noisy):
        """
        Reconstruct clean data from noisy input.
        
        Args:
            x_noisy: np.array shape (N,)
            
        Returns:
            np.array shape (N,) reconstructed (denoised) data
        """
        self.model.eval()
        x_tensor = torch.tensor(x_noisy, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            recon = self.model(x_tensor).cpu().numpy().flatten()
        return recon

def hybrid_residual_corrector(x_true, x_recon, m=23, epsilon=0.3):
    log_x_true, log_residual, residual = compute_log_residuals(x_true, x_recon)
    y_mid, y_upper, y_lower, x_step = compute_rule_based_residual_magnitude(log_x_true, m, epsilon)
    model = ResidualSignModel()
    # model.fit(log_x_true, np.sign(residual))  # Uncomment when implemented
    sign_pred = model.predict_sign(log_x_true)
    reconstructed_residual = sign_pred * (y_upper - y_lower) / 2
    x_recon_corrected_log = log_x_true + reconstructed_residual
    return {
        "log_x_true": log_x_true,
        "log_residual": log_residual,
        "x_step": x_step,
        "y_upper": y_upper,
        "y_lower": y_lower,
        "residual_pred": reconstructed_residual,
        "x_recon_corrected_log": x_recon_corrected_log,
    }

def main():
    branch = "AnalysisElectronsAuxDyn"
    varnames = ["pt"]  # or ["pt", "eta", "phi"], etc.

    batch_size = 1000
    dataloader = get_dataloader(branch=branch, varnames=varnames, batch_size=batch_size, shuffle=False)

    # Grab one batch from dataloader
    # for x_compressed, residual_tensor in dataloader:
    #     # x_true and residual shape: (batch_size, num_vars)
    #     # We'll process only the first variable for this example
    #     x_compressed = x_compressed[:, 0].numpy()
    #     x_recon = x_compressed + residual_tensor[:, 0].numpy()
    #     plot_log_residual_contour(x_compressed, x_recon, gmm=None, varname="pt")
    #     break  # only first batch


    # module = Autoencoder1D(hidden_dim=hidden_dim).to(self.device)
    module = ConvAutoencoder1D()
    model = ResidualSignModel(module, noise_std=0.01, device='cuda' if torch.cuda.is_available() else 'cpu')
    model.fit(dataloader, epochs=1)
    # result = hybrid_residual_corrector(x_true, x_recon)
    # plot_rule_based_bounds(
    #     result["log_x_true"],
    #     result["log_residual"],
    #     result["y_upper"],
    #     result["y_lower"],
    #     result["x_step"],
    #     m=23
    # )

if __name__ == "__main__":
    main()
