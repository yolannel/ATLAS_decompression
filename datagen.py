import os
import numpy as np
import uproot
import awkward as ak
from torch.utils.data import Dataset, DataLoader
import torch

# === Config paths ===
BASE_COMPRESSED_REAL = "/eos/user/y/yolanney/compressed_files/real"
BASE_COMPRESSED_SIM  = "/eos/user/y/yolanney/compressed_files/sim"
UNCOMPRESSED_REAL = [
    "/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/ASG/DAOD_PHYSLITE/p6479/data18_13TeV.00348885.physics_Main.deriv.DAOD_PHYSLITE.r13286_p4910_p6479/DAOD_PHYSLITE.41578717._000256.pool.root.1",
    "/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/ASG/DAOD_PHYSLITE/p6482/data23_13p6TeV.00456749.physics_Main.deriv.DAOD_PHYSLITE.r15774_p6304_p6482/DAOD_PHYSLITE.41588921._000002.pool.root.1"
]

UNCOMPRESSED_SIM = [
    "/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/ASG/DAOD_PHYSLITE/p6490/mc20_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_PHYSLITE.e6337_s3681_r13167_r13146_p6490/DAOD_PHYSLITE.41651753._000007.pool.root.1",
    "/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/ASG/DAOD_PHYSLITE/p6491/mc23_13p6TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.deriv.DAOD_PHYSLITE.e8514_e8528_s4162_s4114_r15540_r15516_p6491/DAOD_PHYSLITE.41633384._000941.pool.root.1"
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
    def __init__(self, file_pairs, branch, varnames):
        """
        Args:
            file_pairs: list of (uncompressed_path, compressed_path, label)
            branch: root tree branch prefix (e.g. "AnalysisElectronsAuxDyn")
            varnames: list of variable names (e.g. ["pt", "eta"])
        """
        self.branch = branch
        self.varnames = varnames if isinstance(varnames, (list, tuple)) else [varnames]
        self.data = []

        for orig_path, comp_path, label in file_pairs:
            self._load_pair(orig_path, comp_path)

        self.data = np.concatenate(self.data, axis=0)

    def _load_pair(self, orig_path, comp_path):
        try:
            tree_orig = uproot.open({orig_path: "CollectionTree"})
            tree_comp = uproot.open({comp_path: "CollectionTree"})
        except Exception as e:
            print(f"Failed to open {orig_path} or {comp_path}: {e}")
            return

        all_true = []
        all_recon = []

        for var in self.varnames:
            full_name = f"{self.branch}.{var}"
            x_true = ak.flatten(tree_orig[full_name].array()).to_numpy()
            x_recon = ak.flatten(tree_comp[full_name].array()).to_numpy()

            mask = (np.abs(x_true) > 0) & np.isfinite(x_true) & np.isfinite(x_recon)

            all_true.append(x_true[mask])
            all_recon.append(x_recon[mask])

        x_true_all = np.stack(all_true, axis=-1)
        x_recon_all = np.stack(all_recon, axis=-1)
        residual = x_true_all - x_recon_all

        # Store x_recon (compressed) and residual
        self.data.append(np.concatenate([x_recon_all, residual], axis=1))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data[idx]
        n = len(self.varnames)
        x_recon = item[:n]
        res = item[n:]
        return torch.tensor(x_recon, dtype=torch.float32), torch.tensor(res, dtype=torch.float32)

# === Helper to Create DataLoader ===
def get_dataloader(branch, varnames, batch_size=256, shuffle=True):
    real_pairs = create_file_pairs("real")
    sim_pairs  = create_file_pairs("sim")
    all_pairs = real_pairs + sim_pairs

    dataset = ROOTResidualDataset(all_pairs, branch=branch, varnames=varnames)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

