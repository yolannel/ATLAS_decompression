"""
bit_inpaint_transformer.py

Small masked-transformer for inpainting zeroed mantissa bits in float32 bitstreams.
Includes helpers for float<->bits and prefix min/max pruning for constrained beam search.

Requirements:
    - Python 3.8+
    - PyTorch (tested with 1.10+)
    - numpy
    - tqdm (optional)
"""

import math
import struct
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from datagen import ROOTBitstreamDataset, create_file_pairs
# ------------------------
# Helpers: float <-> bits
# ------------------------
def float32_to_uint32_bits(x: float) -> int:
    """Return 32-bit unsigned int representation of float32 scalar."""
    return struct.unpack("<I", struct.pack("<f", np.float32(x)))[0]

def uint32_bits_to_float32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", np.uint32(u)))[0]

def float32_to_bits_list(x: float) -> List[int]:
    """Return list of 32 bits (MSB first) for a float32 scalar."""
    u = float32_to_uint32_bits(x)
    bits = [(u >> (31 - i)) & 1 for i in range(32)]
    return bits

def bits_list_to_uint32(bits: List[int]) -> int:
    if len(bits) == 1 and hasattr(bits[0], '__len__') and len(bits[0]) == 32:
        bits = bits[0]

    assert len(bits) == 32
    u = 0
    for i, b in enumerate(bits):
        u |= (int(b) & 1) << (31 - i)
    return u

def bits_list_to_float32(bits: List[int]) -> float:
    u = bits_list_to_uint32(bits)
    return uint32_bits_to_float32(u)

# convenience: extract sign/exponent/mantissa lists
def decompose_bits(bits: List[int]):
    sign = bits[0]
    exponent_bits = bits[1:9]   # 8 bits
    mantissa_bits = bits[9:]    # 23 bits
    return sign, exponent_bits, mantissa_bits

def exponent_bits_to_int(exp_bits: List[int]) -> int:
    e = 0
    for i, b in enumerate(exp_bits):
        e = (e << 1) | int(b)
    return e

# ------------------------
# Mantissa prefix min/max
# ------------------------
def mantissa_fraction_from_bits(bits23: List[int]) -> float:
    # bits23: list of length 23, MSB first (bit for 2^{-1}, 2^{-2}, ...)
    frac = 0.0
    for i, b in enumerate(bits23):
        frac += (int(b) * (2 ** (-(i + 1))))
    return 1.0 + frac  # normalized mantissa in [1,2) for normal numbers

def prefix_min_max_fraction(prefix_bits: List[int], total=23) -> Tuple[float, float]:
    k = len(prefix_bits)
    frac_prefix = 0.0
    for i, b in enumerate(prefix_bits):
        frac_prefix += (int(b) * (2 ** (-(i + 1))))
    # remaining max is sum_{i=k..total-1} 2^{-(i+1)} = 2^{-k} - 2^{-total}
    if k >= total:
        rem_max = 0.0
    else:
        rem_max = (2 ** (-k)) - (2 ** (-total))
    min_frac = 1.0 + frac_prefix
    max_frac = 1.0 + frac_prefix + rem_max
    return min_frac, max_frac

def prefix_min_max_value_for_float(sign: int, exp_biased: int, prefix_bits: List[Optional[int]], fixed_bits_map: Dict[int,int]) -> Tuple[float,float]:
    """
    Compute minimum and maximum possible float value given:
      - sign: 0 or 1
      - exp_biased: biased exponent (0..255)
      - prefix_bits: list of predicted bits in the order of mantissa positions that are already assigned (prefix)
      - fixed_bits_map: dictionary mapping mantissa index (0..22) to fixed bit (0/1) for unmasked positions
    NOTE: we'll construct the partial mantissa from fixed bits + prefix bits placed into masked positions in order.
    """
    # Build mantissa bits array with None for unknowns
    mant = [None] * 23
    # insert fixed bits
    for idx, v in fixed_bits_map.items():
        mant[idx] = int(v)
    # fill prefix (which corresponds to masked positions in increasing index order)
    prefix_positions = [i for i in range(23) if i not in fixed_bits_map]
    for i, b in enumerate(prefix_bits):
        mant[prefix_positions[i]] = int(b)
    # now compute min and max by setting remaining None bits to 0 or 1
    min_bits = [0 if (mb is None) else int(mb) for mb in mant]
    max_bits = [1 if (mb is None) else int(mb) for mb in mant]
    min_frac = mantissa_fraction_from_bits(min_bits)
    max_frac = mantissa_fraction_from_bits(max_bits)
    e = exp_biased - 127
    sign_mul = -1.0 if sign == 1 else 1.0
    # handle subnormals? Simplify: assume normalized numbers for exponents in [1,254]. If exponent 0 it's subnormal; handle separately below.
    if exp_biased == 0:
        # subnormal: value = (-1)^S * 0.F * 2^{1-127} but mantissa semantics differ. Here we approximate conservatively.
        # For subnormals, min and max are small positive numbers; we'll still compute using fractional part without implicit 1.
        # Recompute fractions without the leading 1:
        def frac_from_bits_subnormal(bits23):
            s = 0.0
            for i, b in enumerate(bits23):
                s += int(b) * 2 ** (-(i + 1))
            return s
        min_frac_s = frac_from_bits_subnormal(min_bits)
        max_frac_s = frac_from_bits_subnormal(max_bits)
        base = 2 ** (-(126))  # 2^{1-127}
        min_val = sign_mul * (min_frac_s * base)
        max_val = sign_mul * (max_frac_s * base)
        return min_val, max_val
    # normal numbers:
    min_val = sign_mul * (min_frac * (2.0 ** e))
    max_val = sign_mul * (max_frac * (2.0 ** e))
    # ensure min <= max for negative sign case (when sign=-1 the numeric order flips)
    if sign == 1:
        # negative numbers: min_val <= max_val but both negative; reorder so min<max (more negative -> smaller)
        if min_val > max_val:
            min_val, max_val = max_val, min_val
    else:
        if min_val > max_val:
            min_val, max_val = max_val, min_val
    return min_val, max_val

def intervals_intersect(a: Tuple[float,float], b: Tuple[float,float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])

# ------------------------
# Toy Dataset
# ------------------------
class FloatBitsDataset(Dataset):
    """
    Dataset that yields (bits_tokens, mask_positions, sign, exponent_biased, float_value)
    bits_tokens: 32 tokens: sign(0/1), exponent 8 bits, mantissa 23 bits -> but mantissa masked positions set to MASK_TOKEN (2)
    mask_positions: list of indices in mantissa (0..22) that are masked (we will represent mantissa positions relative to mantissa start)
    """
    MASK_TOKEN = 2

    def __init__(self, floats: np.ndarray, mask_indices: List[int]):
        """
        floats: numpy array of float32 values
        mask_indices: list of mantissa bit indices (0..22) that are zeroed in compressed stream (to simulate)
        """
        self.floats = floats.astype(np.float32)
        self.mask_indices = set(mask_indices)

    def __len__(self):
        return len(self.floats)

    def __getitem__(self, idx):
        x = float(self.floats[idx])
        bits = float32_to_bits_list(x)  # 32 bits MSB first
        sign, exponent_bits, mantissa_bits = decompose_bits(bits)
        exponent_int = exponent_bits_to_int(exponent_bits)
        # Build tokens: tokens[0]=sign, tokens[1:9]=exponent bits, tokens[9:32]=mantissa tokens
        tokens = [sign] + exponent_bits + mantissa_bits[:]  # ints 0/1
        # For training we want to simulate compressed stream: masked mantissa -> set to MASK_TOKEN
        tokens_masked = tokens.copy()
        fixed_bits_map = {}
        mask_positions = []
        for j in range(23):
            global_idx = 9 + j
            if j in self.mask_indices:
                tokens_masked[global_idx] = self.MASK_TOKEN
                mask_positions.append(j)
            else:
                fixed_bits_map[j] = int(tokens[global_idx])
        sample = {
            "tokens_masked": np.array(tokens_masked, dtype=np.int64),
            "tokens_true": np.array(tokens, dtype=np.int64),
            "mask_positions": np.array(mask_positions, dtype=np.int64),  # mantissa positions masked
            "sign": int(sign),
            "exponent_biased": int(exponent_int),
            "fixed_bits_map": fixed_bits_map,
            "float_value": x
        }
        return sample

# Collate to tensors
def collate_fn(batch):
    # All sequences length 32
    batch_tokens_masked = np.stack([b["tokens_masked"] for b in batch], axis=0)  # (B,32)
    batch_tokens_true = np.stack([b["tokens_true"] for b in batch], axis=0)
    sign = np.array([b["sign"] for b in batch], dtype=np.int64)
    exponent_biased = np.array([b["exponent_biased"] for b in batch], dtype=np.int64)
    float_values = np.array([b["float_value"] for b in batch], dtype=np.float32)
    fixed_bits_maps = [b["fixed_bits_map"] for b in batch]
    mask_positions = [b["mask_positions"] for b in batch]
    return {
        "tokens_masked": torch.tensor(batch_tokens_masked, dtype=torch.long),
        "tokens_true": torch.tensor(batch_tokens_true, dtype=torch.long),
        "sign": torch.tensor(sign, dtype=torch.long),
        "exponent_biased": torch.tensor(exponent_biased, dtype=torch.long),
        "float_values": torch.tensor(float_values, dtype=torch.float32),
        "fixed_bits_maps": fixed_bits_maps,
        "mask_positions": mask_positions
    }

# ------------------------
# Model: small masked transformer
# ------------------------
class SmallMaskedTransformer(nn.Module):
    """
    A tiny Transformer encoder that predicts bits per position.
    Vocabulary: {0,1, MASK_TOKEN}
    We'll embed tokens, add position embedding, and add conditioning from sign/exponent via projected vector added to each token embedding.
    """
    def __init__(self, seq_len=32, d_model=192, nhead=6, num_layers=4, dim_feedforward=512, dropout=0.1, mask_token=2):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = 3  # 0,1,MASK
        self.mask_token = mask_token
        self.d_model = d_model

        self.token_emb = nn.Embedding(self.vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        # conditioning projection for exponent and sign
        self.cond_proj = nn.Linear(8 + 1, d_model)  # exponent one-hot (or bits) + sign scalar
        # Alternatively you could use exponent as scalar embedding but using bits is easy: 8 exponent bits as 0/1 input

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Linear(d_model, 2)  # predict bit 0/1 logits for each position (we'll ignore sign/exponent positions when computing loss if desired)

    def forward(self, tokens_masked: torch.FloatTensor, exponent_bits: torch.FloatTensor, sign_vec: torch.FloatTensor):
        """
        tokens_masked: (B, seq_len) ints in {0,1,2}
        exponent_bits: (B, 8) floats (0/1) representing exponent bits
        sign_vec: (B, 1) floats (0/1)
        returns logits: (B, seq_len, 2)
        """
        B, L = tokens_masked.shape
        assert L == self.seq_len
        tok_e = self.token_emb(tokens_masked)  # (B,L,d)
        pos_ids = torch.arange(L, device=tokens_masked.device).unsqueeze(0).expand(B, L)
        pos_e = self.pos_emb(pos_ids)  # (B,L,d)
        cond_in = torch.cat([exponent_bits, sign_vec], dim=1)  # (B,9)
        cond_proj = self.cond_proj(cond_in)  # (B,d)
        cond_proj = cond_proj.unsqueeze(1).expand(B, L, self.d_model)  # add to each position
        x = tok_e + pos_e + cond_proj
        # Transformer encoder
        x = self.encoder(x)  # (B,L,d)
        logits = self.output_head(x)  # (B,L,2)
        return logits

# ------------------------
# Training utilities
# ------------------------
def make_exponent_bits_tensor(exp_biased_array: np.ndarray) -> torch.FloatTensor:
    # convert array of biased exponents (B,) to bit tensors (B,8)
    B = exp_biased_array.shape[0]
    arr = np.zeros((B, 8), dtype=np.float32)
    for i, e in enumerate(exp_biased_array):
        for j in range(8):
            arr[i, 7 - j] = (e >> j) & 1
    return torch.from_numpy(arr)

# def get_mantissa_mask(n_vars, keep_bits=10):
#     """
#     Creates a mask over 32-bit floats:
#     - sign (1) and exponent (8) always known -> mask = 0
#     - mantissa[0:keep_bits] known -> mask = 0
#     - mantissa[keep_bits:23] unknown -> mask = 1
#     """
#     mask = torch.zeros((n_vars, 32), dtype=torch.float32)
#     mask[:, 9+keep_bits+1:] = 1  # careful with indexing
#     # More explicit:
#     # sign + exponent = 9 bits
#     # mantissa[0:keep_bits] = known
#     # mantissa[keep_bits:23] = unknown
#     start = 9 + keep_bits
#     mask[:, start:32] = 1
#     return mask

def train_one_epoch(model, optimizer, dataloader, device, mask_token=2, print_every=100):
    model.train()
    total_loss = 0.0
    nsteps = 0

    for batch_idx, (tokens_masked, tokens_true) in enumerate(dataloader):
        # tokens_masked, tokens_true: (B, 32)
        # print(tokens_masked.shape, tokens_true.shape)
        tokens_masked = tokens_masked.squeeze().to(device)
        tokens_true = tokens_true.squeeze().to(device)

        # Extract sign and exponent bits directly from tokens_true
        sign = tokens_true[:, 0].unsqueeze(1).float()           # (B,1)
        exponent_bits = tokens_true[:, 1:9]                     # (B,8)
        
        # Forward pass
        logits = model(tokens_masked, exponent_bits, sign)      # (B,L,2)

        # --- Build effective mask for loss ---
        # Only positions where:
        # 1) the token is masked (tokens_masked == mask_token)
        # 2) the bit index is in the mantissa (>=9)
        truncated_bits_start = 22  # index of first truncated bit
        effective_mask = torch.zeros_like(tokens_true, dtype=torch.bool)
        effective_mask[:, truncated_bits_start:] = 1


        # Flatten tensors
        logits_flat = logits.view(-1, 2)           # (B*L,2)
        target_flat = tokens_true.view(-1)         # (B*L)
        mask_flat = effective_mask.view(-1)        # (B*L)

        # Select only masked mantissa positions
        logits_sel = logits_flat[mask_flat]
        target_sel = target_flat[mask_flat]

        # Compute loss (ensure target_sel is torch.long)
        loss = F.cross_entropy(logits_sel, target_sel.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        nsteps += 1

        if (batch_idx + 1) % print_every == 0:
            print(f"[train] step {batch_idx+1}/{len(dataloader)} loss={total_loss/nsteps:.4e}")

    return total_loss / max(1, nsteps)

# ------------------------
# Constrained beam search with prefix pruning
# ------------------------
def constrained_beam_search(model, comp_bits: torch.Tensor, orig_bits: torch.Tensor,
                            kept_m: int, allowed_interval: Tuple[float,float], device,
                            beam_width: int = 32, mask_token: int = 2):
    """
    Constrained beam search to inpaint truncated mantissa bits of a float32.

    Args:
        model: trained bit-level transformer
        comp_bits: tensor of shape (32,) of compressed/truncated float bits
        orig_bits: tensor of shape (32,) of original float bits (for sign/exponent extraction)
        kept_m: number of kept mantissa bits in comp_bits
        allowed_interval: (min_val, max_val) allowed float value range
        device: torch device
        beam_width: beam search width
        mask_token: integer used to mark masked bits (default 2)
    Returns:
        List of dicts: {"bits": full 32-bit list, "logprob": total log-prob, "value": float32}
    """
    model.eval()
    # --- Extract sign & exponent ---
    if len(orig_bits) == 1 and hasattr(orig_bits[0], '__len__') and len(orig_bits[0]) == 32:
        orig_bits = orig_bits[0]
    
    if len(comp_bits) == 1 and hasattr(comp_bits[0], '__len__') and len(comp_bits[0]) == 32:
        comp_bits = comp_bits[0]

    sign = int(orig_bits[0].item())
    exponent_bits_list = [int(b.item()) for b in orig_bits[1:9]]
    exponent_biased = exponent_bits_to_int(exponent_bits_list)

    # --- Build masked input ---
    tokens_masked_np = comp_bits.cpu().numpy().copy()
    masked_positions = list(range(kept_m, 23))  # mantissa bits to inpaint
    for i in masked_positions:
        tokens_masked_np[9+i] = mask_token

    seq_len = len(tokens_masked_np)
    beam = [([], 0.0)]  # (prefix_bits_list, logprob)

    for step_idx in range(len(masked_positions)):
        new_beam = []
        for prefix_bits, lp in beam:
            for bit in (0,1):
                candidate_prefix = prefix_bits + [bit]

                # Construct tokens for model
                tokens_try = tokens_masked_np.copy()
                for i_b, bval in enumerate(candidate_prefix):
                    global_idx = 9 + masked_positions[i_b]
                    tokens_try[global_idx] = int(bval)

                # Model forward
                with torch.no_grad():
                    tokens_tensor = torch.tensor(tokens_try, dtype=torch.long, device=device).unsqueeze(0)
                    exp_bits_tensor = torch.tensor([exponent_bits_list], dtype=torch.long, device=device)
                    sign_tensor = torch.tensor([[sign]], dtype=torch.float32, device=device)
                    logits = model(tokens_tensor, exp_bits_tensor, sign_tensor)  # (1,32,2)

                    # Log-prob of the current bit
                    next_masked_pos = masked_positions[len(candidate_prefix)-1]
                    global_pos = 9 + next_masked_pos
                    logp = F.log_softmax(logits[0, global_pos], dim=0)[bit].item()
                    new_lp = lp + logp

                # Check interval pruning
                full_mant = comp_bits[9:].cpu().numpy().tolist()  # start from compressed mantissa
                for i_b, bval in enumerate(candidate_prefix):
                    full_mant[masked_positions[i_b]] = int(bval)
                bits32 = [sign] + exponent_bits_list + full_mant
                val = bits_list_to_float32(bits32)
                if val < allowed_interval[0] or val > allowed_interval[1]:
                    continue  # prune

                new_beam.append((candidate_prefix, new_lp))

        if len(new_beam) == 0:
            break  # all candidates pruned; rare
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]

    # --- Finalize beams ---
    results = []
    for prefix_bits, lp in beam:
        full_mant = comp_bits[9:].cpu().numpy().tolist()
        for i_b, bval in enumerate(prefix_bits):
            full_mant[masked_positions[i_b]] = int(bval)
        bits32 = [sign] + exponent_bits_list + full_mant
        val = bits_list_to_float32(bits32)
        results.append({"bits": bits32, "logprob": lp, "value": val})

    results.sort(key=lambda r: r["logprob"], reverse=True)
    return results

# ------------------------
# Utilities: simple baselines
# ------------------------
def zero_fill_baseline(tokens_masked_np: np.ndarray):
    t = tokens_masked_np.copy()
    for i in range(9, 32):
        if t[i] == 2:
            t[i] = 0
    return t

def midpoint_baseline(tokens_masked_np: np.ndarray, sign:int, exponent_biased:int, fixed_bits_map: Dict[int,int]):
    # set masked mantissa bits to 1/2? a simple midpoint filling: set all masked bits to 1 (max) and compute average with zeros
    t0 = zero_fill_baseline(tokens_masked_np)
    t1 = tokens_masked_np.copy()
    for i in range(9, 32):
        if t1[i] == 2:
            t1[i] = 1
    # compute values and return whichever is closer to mid of allowed range? For simplicity return zeros fill
    return t0

from tqdm import tqdm

def evaluate_inpainting(model, dataset, kept_m=13, device=None, beam_width=64):
    """
    Evaluate inpainting quality on a dataset.
    
    Args:
        model: trained bit-level transformer
        dataset: dataset returning (comp_bits, orig_bits)
        kept_m: number of kept mantissa bits in comp_bits
        device: torch device
        beam_width: beam width for constrained beam search
        
    Returns:
        dict with MSE/MAE for compressed vs true, top inpainted vs true, and optionally all inpainted avg
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mse_comp_list = []
    mae_comp_list = []
    mse_inpaint_list = []
    mae_inpaint_list = []
    count = 0
    for comp_bits, orig_bits in tqdm(dataset):
        # float values
        if count > 5000:
            break
        compressed_val = bits_list_to_float32(comp_bits.tolist())
        true_val = bits_list_to_float32(orig_bits.tolist())

        # number of truncated bits
        delta = 2 ** (exponent_bits_to_int(orig_bits[1:9].tolist()) - 127 - (23 - kept_m))
        allowed_interval = (true_val - delta, true_val + delta)

        # beam search
        results = constrained_beam_search(
            model,
            comp_bits,
            orig_bits,
            kept_m,
            allowed_interval,
            device,
            beam_width=beam_width
        )

        top_inpainted_val = results[0]["value"]

        # metrics
        mse_comp_list.append((compressed_val - true_val)**2)
        mae_comp_list.append(abs(compressed_val - true_val))
        mse_inpaint_list.append((top_inpainted_val - true_val)**2)
        mae_inpaint_list.append(abs(top_inpainted_val - true_val))
        count += 1

    metrics = {
        "MSE_compressed": np.mean(mse_comp_list),
        "MAE_compressed": np.mean(mae_comp_list),
        "MSE_inpainted_top": np.mean(mse_inpaint_list),
        "MAE_inpainted_top": np.mean(mae_inpaint_list),
    }
    return metrics


# ------------------------
# Example main for toy run
# ------------------------
def example_toy_run():
    # generate synthetic distribution of floats (positive magnitudes, varied exponents)
    N = 20000
    # sample mantissa and exponent uniformly to get variety across exponents
    exponents = np.random.randint(1, 250, size=N)  # avoid denormals and inf/NaN
    mantissa_fracs = np.random.rand(N)  # 0..1
    signs = np.zeros(N, dtype=np.int64)
    # vals = np.array([ ((1.0 + m) * (2.0 ** (e - 127))) for m,e in zip(mantissa_fracs, exponents)], dtype=np.float32)
    # simple mask indices: e.g., zero out lower 10 mantissa bits
    # mask_indices = list(range(13, 23))  # last 10 bits masked
    # dataset = FloatBitsDataset(vals, mask_indices=mask_indices)
    # dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    real_pairs = create_file_pairs("real")
    sim_pairs  = create_file_pairs("sim")
    all_pairs = real_pairs + sim_pairs
    dataloader = DataLoader(
        ROOTBitstreamDataset(file_pairs=all_pairs, branch="AnalysisElectronsAuxDyn", varnames=["pt"]),
        batch_size=64, shuffle=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallMaskedTransformer(seq_len=32, d_model=192, nhead=6, num_layers=4, dim_feedforward=512).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)

    # train for a few epochs
    for epoch in range(6):
        break
        loss = train_one_epoch(model, optimizer, dataloader, device, mask_token=2, print_every=20)
        print(f"Epoch {epoch}: loss {loss:.4e}")

    # pick a sample and run constrained beam search
    # pick a sample
    # pick a single sample from your dataset
    dataset = dataloader.dataset
    metrics = evaluate_inpainting(model, dataset, kept_m=13, device=device, beam_width=32)
    print(metrics)

    comp_bits, orig_bits = dataloader.dataset[123]  # comp_bits, orig_bits: torch tensors (32,)

    # number of kept mantissa bits (e.g., you truncated 10 bits → kept 13 bits)
    kept_m = 23 - 10  # 13

    # allowed interval around the true value (you can use your theoretical bounds)
    true_value = bits_list_to_float32(orig_bits.tolist())
    delta = 2 ** (exponent_bits_to_int(orig_bits[1:9].tolist()) - 127 - (23 - kept_m))
    allowed_interval = (true_value - delta, true_value + delta)

    # run beam search
    results = constrained_beam_search(
        model,
        comp_bits,          # shape (32,)
        orig_bits,          # shape (32,)
        kept_m,
        allowed_interval,
        device,
        beam_width=64
    )

    

    # print top candidates
    for r in results[:8]:
        print("Value:", r["value"], "Logprob:", r["logprob"])

    print("True value:", true_value)


    # Show zero-fill baseline
    z = zero_fill_baseline(comp_bits[0].numpy())
    print("Zero-fill value:", bits_list_to_float32(z.tolist()))

    # Collect inpainted float32 values
    inpainted_values = []
    for r in results:
        val = r["value"]
        if isinstance(val, (list, np.ndarray, torch.Tensor)):
            val = bits_list_to_float32(list(val))
        inpainted_values.append(val)

    original_value = true_value

    # === Plot histogram of generated floats ===
    plt.figure(figsize=(8, 5))
    plt.hist(inpainted_values, bins=100, log=True)
    plt.xscale('log')
    plt.xlabel('Generated Float Value (log scale)')
    plt.ylabel('Count (log scale)')
    plt.title('Distribution of Generated Float Values')
    plt.tight_layout()
    plt.show()

    # === Residuals for top-N candidates ===
    N = min(32, len(inpainted_values))
    residuals = [v - original_value for v in inpainted_values[:N]]
    originals = [original_value] * N

    plt.figure(figsize=(8, 5))
    plt.scatter(originals, residuals, c='blue', label='Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Original Value (base 10)')
    plt.ylabel('Signed Residual (Inpainted - Original)')
    plt.title('Residuals of Inpainted vs Original Floats')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------
# Entry point
# ------------------------
if __name__ == "__main__":
    example_toy_run()
