"""
Release: per-branch R^2 for the r-sweep (Appendix A.2: dominant-branch R^2, median branch R^2 at r = 64).
Branch fits skip the k = 0 mode (Table A.8 in Appendix A.10, from analyze_spectral.py, keeps it).
    python navier_stokes/ablations/ablation_r2_fix.py --save_dir checkpoints/ns_ablations/r_sweep

Original docstring:
Recompute eigenspectrum R² per branch for ablation checkpoints.
================================================================
For each r and viscosity, loads the best checkpoint, computes R² for
EACH of the r eigenvalue branches (sorted by mean Re(λ)), and reports:
  - Best single-branch R²
  - Mean of top-3 branch R²
  - Original dominant-eigenvalue R² (for comparison)

Then regenerates the ablation figure with the corrected metric.

Usage:
    python ablation_r2_fix.py --save_dir ./ablation_results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import matrix_exp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse, os, json, csv
from scipy.stats import linregress

torch.manual_seed(0)
np.random.seed(0)

R_VALUES = [4, 8, 16, 32, 64]

# ============================================================
# Model (identical to ablation_r.py)
# ============================================================

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out = torch.zeros(B, self.w1.shape[1], H, W//2+1, dtype=torch.cfloat, device=x.device)
        out[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.w1)
        out[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.w2)
        return torch.fft.irfft2(out, s=(H, W))


class SpatialEncoder(nn.Module):
    def __init__(self, modes, width, r, T_in):
        super().__init__()
        self.fc0   = nn.Linear(T_in + 2, width)
        self.conv0 = SpectralConv2d(width, width, modes, modes)
        self.conv1 = SpectralConv2d(width, width, modes, modes)
        self.w0    = nn.Conv2d(width, width, 1)
        self.w1    = nn.Conv2d(width, width, 1)
        self.proj  = nn.Conv2d(width, r, 1)

    def forward(self, x, grid):
        x = torch.cat([x, grid], dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        return self.proj(x)


class SpatialDecoder(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(r, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))

    def forward(self, z):
        return self.net(z).squeeze(1)


class SpatialKoopmanLGN(nn.Module):
    def __init__(self, modes, width, r, T_in, H=64, W=64):
        super().__init__()
        self.modes, self.r, self.H, self.W = modes, r, H, W
        self.encoder = SpatialEncoder(modes, width, r, T_in)
        self.decoder = SpatialDecoder(r)
        self.P     = nn.Parameter(torch.randn(r, r) * 0.01)
        self.d     = nn.Parameter(torch.full((r,), -2.0))
        self.alpha = nn.Parameter(torch.full((r,), -3.0))

        k1 = torch.arange(modes, dtype=torch.float32)
        k2 = torch.arange(modes, dtype=torch.float32)
        kk1, kk2 = torch.meshgrid(k1, k2, indexing='ij')
        k2_norm = (kk1**2 + kk2**2) / (modes**2)
        self.register_buffer('k2_norm', k2_norm)

    def get_generators(self):
        S  = (self.P - self.P.T) / 2.0
        d0 = F.softplus(self.d)
        a0 = F.softplus(self.alpha)
        D_diag = d0.unsqueeze(0).unsqueeze(0) + \
                 a0.unsqueeze(0).unsqueeze(0) * self.k2_norm.unsqueeze(-1)
        S_exp = S.unsqueeze(0).unsqueeze(0).expand(self.modes, self.modes, -1, -1).clone()
        D_mat = torch.diag_embed(D_diag)
        return S_exp - D_mat

    def get_max_re_eigenvalue(self):
        L_all = self.get_generators()
        L_flat = L_all.reshape(-1, self.r, self.r)
        eigs = torch.linalg.eigvals(L_flat)
        return eigs.real.max().item()

    def get_branch_r2(self):
        """
        Compute R² for each eigenvalue branch separately.

        At each mode k, compute r eigenvalues, sort by real part (descending).
        This gives r branches across M² modes. Compute R² of Re(λ) vs |k|²
        for each branch independently.

        Returns dict with:
          - branch_r2: list of R² per branch (sorted most to least negative mean)
          - best_r2: max R² across branches
          - dominant_r2: R² of branch 0 (least negative = "dominant")
          - most_dissipative_r2: R² of the most dissipative branch
          - mean_re_r2: R² computed on mean Re(λ) across all branches
        """
        L_all = self.get_generators()  # (M, M, r, r)
        M = self.modes
        r = self.r

        # Collect eigenvalues at each mode
        k2_vals = []
        all_eigs_sorted = []  # (n_modes, r) sorted Re(λ) descending per mode

        for i in range(M):
            for j in range(M):
                if i == 0 and j == 0:
                    continue
                k2 = i**2 + j**2
                L_k = L_all[i, j]
                eigs = torch.linalg.eigvals(L_k)
                # Sort by real part descending (least negative first = "dominant")
                re_sorted, _ = eigs.real.sort(descending=True)
                k2_vals.append(k2)
                all_eigs_sorted.append(re_sorted.detach().cpu().numpy())

        k2_arr = np.array(k2_vals)
        eig_matrix = np.stack(all_eigs_sorted)  # (n_modes, r)

        # R² per branch
        branch_r2 = []
        for b in range(r):
            re_branch = eig_matrix[:, b]
            slope, intercept, r_value, _, _ = linregress(k2_arr, re_branch)
            branch_r2.append(r_value**2)

        # Mean Re(λ) across all branches
        mean_re = eig_matrix.mean(axis=1)
        _, _, r_val_mean, _, _ = linregress(k2_arr, mean_re)
        mean_re_r2 = r_val_mean**2

        # Most dissipative = branch with most negative mean Re(λ) = last branch
        # Dominant = branch 0 (least negative)
        results = {
            'branch_r2': branch_r2,
            'best_r2': max(branch_r2),
            'best_branch_idx': int(np.argmax(branch_r2)),
            'dominant_r2': branch_r2[0],
            'most_dissipative_r2': branch_r2[-1],
            'mean_re_r2': mean_re_r2,
            'top3_mean_r2': float(np.mean(sorted(branch_r2, reverse=True)[:min(3, r)])),
        }
        return results


def analyze_checkpoint(r, tag, save_dir, modes=12, width=32, T_in=10):
    """Load checkpoint and compute branch-wise R²."""
    run_dir = os.path.join(save_dir, f'r{r}_{tag}')
    ckpt_path = os.path.join(run_dir, 'best.pt')

    if not os.path.exists(ckpt_path):
        print(f"  SKIP: {ckpt_path} not found")
        return None

    model = SpatialKoopmanLGN(modes=modes, width=width, r=r, T_in=T_in)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    model.eval()

    with torch.no_grad():
        results = model.get_branch_r2()

    # Load original results for test L2
    json_path = os.path.join(run_dir, 'results.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            orig = json.load(f)
        results['test_l2'] = orig['best_test_l2']
        results['original_r2'] = orig['eigenspectrum_r2']
    else:
        results['test_l2'] = None
        results['original_r2'] = None

    results['r'] = r
    results['tag'] = tag

    # Save updated results
    updated_path = os.path.join(run_dir, 'results_branch_r2.json')
    # Convert numpy types for JSON
    save_results = {k: (float(v) if isinstance(v, (np.floating, float)) else
                        [float(x) for x in v] if isinstance(v, list) else v)
                    for k, v in results.items()}
    with open(updated_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    return results


def generate_figure(all_results, save_dir):
    """Generate corrected ablation figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = {'1e5': '#2196F3', '1e3': '#FF5722'}
    labels = {'1e5': 'ν=1e-5', '1e3': 'ν=1e-3'}

    for tag in ['1e5', '1e3']:
        res_list = [r for r in all_results if r['tag'] == tag]
        res_list.sort(key=lambda x: x['r'])

        if not res_list:
            continue

        rs = [r['r'] for r in res_list]
        l2s = [r['test_l2'] for r in res_list]
        best_r2s = [r['best_r2'] for r in res_list]
        orig_r2s = [r['original_r2'] for r in res_list]

        c = colors[tag]
        lab = labels[tag]

        # Panel A: Test L2 vs r
        axes[0].plot(rs, l2s, 'o-', color=c, linewidth=2, markersize=8, label=lab)
        for rv, lv in zip(rs, l2s):
            if lv is not None:
                axes[0].annotate(f'{lv:.3f}', (rv, lv),
                               textcoords="offset points", xytext=(0, 10),
                               fontsize=8, ha='center', color=c)

        # Panel B: Best-branch R² vs r (solid) + original dominant R² (dashed)
        axes[1].plot(rs, best_r2s, 's-', color=c, linewidth=2, markersize=8, label=f'{lab} (best branch)')
        axes[1].plot(rs, orig_r2s, 's--', color=c, linewidth=1.5, markersize=6, alpha=0.4,
                    label=f'{lab} (dominant only)')
        for rv, bv in zip(rs, best_r2s):
            axes[1].annotate(f'{bv:.3f}', (rv, bv),
                           textcoords="offset points", xytext=(0, 10),
                           fontsize=8, ha='center', color=c)

    axes[0].set_xlabel('Channel Dimension r', fontsize=13)
    axes[0].set_ylabel('Test Relative L2', fontsize=13)
    axes[0].set_title('Prediction Accuracy vs r', fontsize=14, fontweight='bold')
    axes[0].set_xticks(R_VALUES)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Channel Dimension r', fontsize=13)
    axes[1].set_ylabel('R² (Re(λ) vs |k|²)', fontsize=13)
    axes[1].set_title('Kolmogorov Scaling Recovery vs r', fontsize=14, fontweight='bold')
    axes[1].set_xticks(R_VALUES)
    axes[1].set_ylim(0.5, 1.05)
    axes[1].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=8, loc='lower left')
    axes[1].grid(True, alpha=0.3)

    # Panel C: Which branch carries the scaling
    axes[2].set_xlabel('Channel Dimension r', fontsize=13)
    axes[2].set_ylabel('Best Branch Index', fontsize=13)
    axes[2].set_title('Which Branch Carries\nKolmogorov Scaling', fontsize=14, fontweight='bold')
    for tag in ['1e5', '1e3']:
        res_list = [r for r in all_results if r['tag'] == tag]
        res_list.sort(key=lambda x: x['r'])
        if not res_list:
            continue
        rs = [r['r'] for r in res_list]
        best_idx = [r['best_branch_idx'] for r in res_list]
        c = colors[tag]
        lab = labels[tag]
        axes[2].plot(rs, best_idx, 'D-', color=c, linewidth=2, markersize=8, label=lab)
        for rv, bi in zip(rs, best_idx):
            axes[2].annotate(f'{bi}', (rv, bi),
                           textcoords="offset points", xytext=(0, 10),
                           fontsize=9, ha='center', color=c, fontweight='bold')
    axes[2].set_xticks(R_VALUES)
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'fig_ablation_r_corrected.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"\n  Saved: {fig_path}")

    # Print summary table
    print(f"\n{'='*100}")
    print(f"{'r':>4}  {'tag':>4}  {'test_L2':>8}  {'orig_R²':>8}  {'best_R²':>8}  "
          f"{'best_branch':>11}  {'dominant_R²':>11}  {'dissip_R²':>10}  {'mean_R²':>8}")
    print(f"{'-'*100}")
    for res in sorted(all_results, key=lambda x: (x['tag'], x['r'])):
        print(f"{res['r']:>4}  {res['tag']:>4}  {res['test_l2']:>8.4f}  "
              f"{res['original_r2']:>8.3f}  {res['best_r2']:>8.3f}  "
              f"{res['best_branch_idx']:>11}  {res['dominant_r2']:>11.3f}  "
              f"{res['most_dissipative_r2']:>10.3f}  {res['mean_re_r2']:>8.3f}")
    print(f"{'='*100}")

    # Save CSV
    csv_path = os.path.join(save_dir, 'ablation_summary_corrected.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['r', 'viscosity', 'test_l2', 'original_r2', 'best_branch_r2',
                     'best_branch_idx', 'dominant_r2', 'most_dissipative_r2', 'mean_re_r2'])
        for res in sorted(all_results, key=lambda x: (x['tag'], x['r'])):
            w.writerow([res['r'], res['tag'], f"{res['test_l2']:.4f}",
                       f"{res['original_r2']:.3f}", f"{res['best_r2']:.3f}",
                       res['best_branch_idx'], f"{res['dominant_r2']:.3f}",
                       f"{res['most_dissipative_r2']:.3f}", f"{res['mean_re_r2']:.3f}"])
    print(f"  Saved: {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='./ablation_results')
    args = parser.parse_args()

    all_results = []

    for tag in ['1e5', '1e3']:
        for r in R_VALUES:
            print(f"Analyzing r={r} {tag}...")
            res = analyze_checkpoint(r, tag, args.save_dir)
            if res is not None:
                all_results.append(res)

    if all_results:
        generate_figure(all_results, args.save_dir)
    else:
        print("No checkpoints found!")
