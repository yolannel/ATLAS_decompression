import os
import numpy as np
import uproot
import awkward as ak
from torch.utils.data import Dataset, DataLoader
import torch

"""
Datasets and data-loading utilities for learning floating-point
residual corrections from lossy-compressed ATLAS PHYSLITE ROOT data.

This module defines PyTorch Dataset classes that transform matched
pairs of original and compressed ROOT files into machine-learning-
ready inputs and targets.
"""


# === Config paths ===
# BASE_COMPRESSED_REAL = "/eos/user/y/yolanney/compressed_files/real"
# BASE_COMPRESSED_SIM  = "/eos/user/y/yolanney/compressed_files/sim"
# UNCOMPRESSED_REAL = [
#     "/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/ASG/DAOD_PHYSLITE/p6479/data18_13TeV.00348885.physics_Main.deriv.DAOD_PHYSLITE.r13286_p4910_p6479/DAOD_PHYSLITE.41578717._000256.pool.root.1",
#     "/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/ASG/DAOD_PHYSLITE/p6482/data23_13p6TeV.00456749.physics_Main.deriv.DAOD_PHYSLITE.r15774_p6304_p6482/DAOD_PHYSLITE.41588921._000002.pool.root.1"
# ]

# UNCOMPRESSED_SIM = [
#     "/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/ASG/DAOD_PHYSLITE/p6490/mc20_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_PHYSLITE.e6337_s3681_r13167_r13146_p6490/DAOD_PHYSLITE.41651753._000007.pool.root.1",
#     "/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/ASG/DAOD_PHYSLITE/p6491/mc23_13p6TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.deriv.DAOD_PHYSLITE.e8514_e8528_s4162_s4114_r15540_r15516_p6491/DAOD_PHYSLITE.41633384._000941.pool.root.1"
# ]
BASE_COMPRESSED_REAL = "~/data/ATLAS/compressed_files/real"
BASE_COMPRESSED_SIM = "~/data/ATLAS/compressed_files/sim"
UNCOMPRESSED_REAL = [
    "~/data/ATLAS/original_files/real/DAOD_PHYSLITE.41578717._000256.pool.root.1",
    "~/data/ATLAS/original_files/real/DAOD_PHYSLITE.41588921._000002.pool.root.1"
    ]
UNCOMPRESSED_SIM = [
    "~/data/ATLAS/original_files/sim/DAOD_PHYSLITE.41633384._000941.pool.root.1",
    "~/data/ATLAS/original_files/sim/DAOD_PHYSLITE.41651753._000007.pool.root.1"
]


def make_compressed_path(original_path, category, level_tag="dl10"):
    base = BASE_COMPRESSED_REAL if category == "real" else BASE_COMPRESSED_SIM
    filename = os.path.basename(original_path)

    # Strip final `.1` if present
    if filename.endswith(".1"):
        filename = filename[:-2]

    compressed_filename = f"{filename}_{level_tag}_compressed.root"
    return os.path.join(base, compressed_filename)

def create_file_pairs(data_category):
    uncompressed_files = UNCOMPRESSED_REAL if data_category == "real" else UNCOMPRESSED_SIM
    pairs = []
    for path in uncompressed_files:
        comp = make_compressed_path(path, data_category.lower())
        pairs.append((path, comp, data_category))
    return pairs

# === Torch Dataset ===
class ROOTResidualDataset(Dataset):

    """
    PyTorch Dataset for learning floating-point residual corrections
    from lossy-compressed ATLAS PHYSLITE ROOT data.

    Each sample is constructed from a matched pair of:
    - original (uncompressed) physics values
    - compressed (lossy) physics values

    The dataset returns:
    - input: compressed floating-point values
    - target: residuals (original - compressed)

    Learning residuals instead of absolute values stabilizes
    training and preserves the statistical structure of
    underlying physics distributions.

    Supported modes:
    - Eventwise mode (default): one physics object per sample
    - Full sample mode: all events from one ROOT file per sample
    - Histogram mode: distribution-level residual learning
    """

    def __init__(self, file_pairs, branch, varnames, 
                 histogram_mode=False, bins=1000, range=None, full_sample_mode=False):
        """
        Args:
            file_pairs: list of (uncompressed_path, compressed_path, label)
            branch: root tree branch prefix (e.g. "AnalysisElectronsAuxDyn")
            varnames: list of variable names (e.g. ["pt", "eta"])
            histogram_mode: if True, dataset stores histograms instead of eventwise samples
            bins: number of histogram bins (int or sequence for np.histogram)
            range: optional histogram range (tuple or list of tuples per var)
            full_sample_mode: if True, each __getitem__ returns all events from one file pair
        """
        self.branch = branch
        self.varnames = varnames if isinstance(varnames, (list, tuple)) else [varnames]
        self.histogram_mode = histogram_mode
        self.bins = bins
        self.range = range
        self.full_sample_mode = full_sample_mode

        self.samples = []   # store one entry per file
        for orig_path, comp_path, label in file_pairs:
            sample = self._load_pair(orig_path, comp_path)
            if sample is not None:
                self.samples.append(sample)

        if not self.histogram_mode and not self.full_sample_mode:
            # flatten all events into one array
            self.samples = np.concatenate(self.samples, axis=0)

    def _load_pair(self, orig_path, comp_path):
        """
        Load a matched pair of original and compressed ROOT files and
        convert them into machine-learning-ready numerical arrays.

        This function:
        - Opens synchronized ROOT trees from uncompressed and compressed files
        - Extracts selected physics variables
        - Applies basic validity masking
        - Computes residuals (original - compressed)
        - Returns either event-level or distribution-level samples

        Args:
            orig_path (str): Path to the uncompressed ROOT file
            comp_path (str): Path to the compressed ROOT file

        Returns:
            np.ndarray or None:
                - Event / full-sample mode: concatenated array of
                [compressed values | residuals]
                - Histogram mode: concatenated array of
                [compressed histogram | histogram residual]
                - None if files cannot be opened
        """

        try:
            tree_orig = uproot.open({orig_path: "CollectionTree"})
            tree_comp = uproot.open({comp_path: "CollectionTree"})
        except Exception as e:
            print(f"Failed to open {orig_path} or {comp_path}: {e}")
            return None

        all_true, all_recon = [], []
        for var in self.varnames:
            full_name = f"{self.branch}.{var}"
            x_true = ak.flatten(tree_orig[full_name].array()).to_numpy()
            x_recon = ak.flatten(tree_comp[full_name].array()).to_numpy()

            if len(x_true) != len(x_recon):
                raise ValueError(
                    f"Length mismatch for {full_name}: "
                    f"{len(x_true)} (original) vs {len(x_recon)} (compressed)"
                )

            mask = (np.abs(x_true) > 0) & np.isfinite(x_true) & np.isfinite(x_recon)
            if np.count_nonzero(mask) == 0:
                raise ValueError(
                    f"All values masked for variable {full_name} "
                    "Check data validity or masking criteria"
                )

            all_true.append(x_true[mask])
            all_recon.append(x_recon[mask])

        x_true_all = np.stack(all_true, axis=-1)
        x_recon_all = np.stack(all_recon, axis=-1)

        if (not np.all (np.isfinite(x_true_all))) or (not np.all(np.isfinite(x_recon_all))):
            raise ValueError("Non-finite values detected after masking")


        if self.histogram_mode:
            # log-spaced histogram bins
            if self.range is not None:
                low, high = self.range
                if low <= 0:
                    raise ValueError("Log-spaced bins require range > 0.")
                bin_edges = np.logspace(np.log10(low), np.log10(high), self.bins + 1)
            else:
                low = np.min(x_true_all[x_true_all > 0])
                high = np.max(x_true_all)
                self.range = (low, high)
                bin_edges = np.logspace(np.log10(low), np.log10(high), self.bins + 1)

            hists_true, hists_recon = [], []
            for i in range(len(self.varnames)):
                h_true, _ = np.histogram(x_true_all[:, i], bins=bin_edges, density=True)
                h_recon, _ = np.histogram(x_recon_all[:, i], bins=bin_edges, density=True)
                hists_true.append(h_true)
                hists_recon.append(h_recon)

            hists_true = np.concatenate(hists_true, axis=0)
            hists_recon = np.concatenate(hists_recon, axis=0)

            residual_hist = hists_true - hists_recon
            return np.concatenate([hists_recon, residual_hist], axis=0)

        else:
            residual = x_true_all - x_recon_all
            return np.concatenate([x_recon_all, residual], axis=1)

    def __len__(self):
        if self.histogram_mode or self.full_sample_mode:
            return len(self.samples)
        else:
            return self.samples.shape[0]

    def __getitem__(self, idx):
        if self.histogram_mode:
            n = len(self.varnames) * self.bins
            item = self.samples[idx]
            x_recon_hist, res_hist = item[:n], item[n:]
            return torch.tensor(x_recon_hist, dtype=torch.float32), torch.tensor(res_hist, dtype=torch.float32)

        elif self.full_sample_mode:
            # return the full sample (all events) for this file pair
            item = self.samples[idx]
            n = len(self.varnames)
            x_recon, res = item[:, :n], item[:, n:]
            return torch.tensor(x_recon, dtype=torch.float32), torch.tensor(res, dtype=torch.float32)

        else:
            # eventwise
            item = self.samples[idx]
            n = len(self.varnames)
            x_recon, res = item[:n], item[n:]
            return torch.tensor(x_recon, dtype=torch.float32), torch.tensor(res, dtype=torch.float32)



# === Helper to Create DataLoader ===
def get_dataloader(branch, varnames, hist=False, full_sample_mode=False, batch_size=256, shuffle=True, range=None):
    """
    Convenience helper to construct a DataLoader for the residual dataset.

    Combines real and simulated file pairs and returns a PyTorch
    DataLoader with the requested sampling mode.
    """

    real_pairs = create_file_pairs("real")
    sim_pairs  = create_file_pairs("sim")
    all_pairs = real_pairs + sim_pairs

    dataset = ROOTResidualDataset(all_pairs, branch=branch, varnames=varnames, histogram_mode=hist, full_sample_mode=full_sample_mode, range=range)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def float_to_bitarray(x: np.ndarray) -> np.ndarray:
    """Convert float32 array into (n, 32) binary array."""
    # ensure float32 dtype
    x = x.astype(np.float32)
    # view as uint32, then bytes
    as_uint = x.view(np.uint32)
    # unpack to bits
    bits = np.unpackbits(as_uint.view(np.uint8)).reshape(-1, 32)
    return bits

class ROOTBitstreamDataset(ROOTResidualDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = self.samples[idx]
        n = len(self.varnames)

        # compressed & residual
        x_comp, res = item[:n], item[n:]
        x_orig = x_comp + res

        # convert to bit arrays
        comp_bits = float_to_bitarray(x_comp)
        orig_bits = float_to_bitarray(x_orig)

        return (
            torch.tensor(comp_bits, dtype=torch.int32),
            torch.tensor(orig_bits, dtype=torch.int32),
        )
