"""Roll out LGN-KM and FNO on the nu = 1e-5 benchmark: per-step error and 200-step energy.

Paper: Section 4.1, Fig. 5a (per-step relative L2, 200 test trajectories, t = 1..10) and
Fig. 5b (decoded mean enstrophy over 200 steps, 5 test trajectories, no ground truth).

Usage:
    python long_horizon/fig5ab_rollout.py --data_1e5 data/ns/NavierStokes_V1e-5_N1200_T20.mat \
        --ckpt_koopman_1e5 checkpoints/ns/lgnkm_nu1e5_s0.pt --ckpt_fno_1e5 checkpoints/ns/fno_nu1e5.pt

Writes long_horizon_{per_step,energy,summary}.csv and fig_long_horizon.png to --save_dir.
The paper's output is in long_horizon/results/fig5ab/.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import matrix_exp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse, os

torch.manual_seed(0)
np.random.seed(0)

# ============================================================
# Model definitions (shared by all analyses)
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
    def __init__(self, modes=12, width=32, r=32, T_in=10, H=64, W=64):
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
        D_diag = d0[None, None, :] + a0[None, None, :] * self.k2_norm[:, :, None]
        S_exp = S[None, None, :, :].expand(self.modes, self.modes, -1, -1).clone()
        D_mat = torch.diag_embed(D_diag)
        return S_exp - D_mat

    def get_generators_rescaled(self, scale_factor):
        """Return generators with D_k rescaled by scale_factor (for OOD transfer)."""
        S  = (self.P - self.P.T) / 2.0
        d0 = F.softplus(self.d) * scale_factor
        a0 = F.softplus(self.alpha) * scale_factor
        D_diag = d0[None, None, :] + a0[None, None, :] * self.k2_norm[:, :, None]
        S_exp = S[None, None, :, :].expand(self.modes, self.modes, -1, -1).clone()
        D_mat = torch.diag_embed(D_diag)
        return S_exp - D_mat

    def propagate(self, z0_ft, t, dt=1.0, L_override=None):
        L_all = L_override if L_override is not None else self.get_generators()
        eL = matrix_exp(L_all * t * dt)
        z = z0_ft.permute(0, 2, 3, 1).unsqueeze(-1)
        z_t = (eL.unsqueeze(0).to(torch.cfloat) @ z).squeeze(-1)
        return z_t.permute(0, 3, 1, 2)

    def forward(self, x, grid, T_out, dt=1.0, L_override=None):
        z0    = self.encoder(x, grid)
        z0_ft = torch.fft.rfft2(z0)
        z0_ft_trunc = z0_ft[:, :, :self.modes, :self.modes]
        preds = []
        for t in range(1, T_out + 1):
            zt_ft_trunc = self.propagate(z0_ft_trunc, t, dt, L_override)
            zt_ft = z0_ft.clone()
            zt_ft[:, :, :self.modes, :self.modes] = zt_ft_trunc
            z_t = torch.fft.irfft2(zt_ft, s=(self.H, self.W))
            preds.append(self.decoder(z_t))
        return torch.stack(preds, dim=1)


class FNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=20, T_in=10):
        super().__init__()
        self.modes1, self.modes2, self.width, self.T_in = modes1, modes2, width, T_in
        self.fc0 = nn.Linear(T_in + 2, width)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, grid):
        x = torch.cat([x, grid], dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        x1 = self.conv0(x); x2 = self.w0(x); x = F.gelu(x1 + x2)
        x1 = self.conv1(x); x2 = self.w1(x); x = F.gelu(x1 + x2)
        x1 = self.conv2(x); x2 = self.w2(x); x = F.gelu(x1 + x2)
        x1 = self.conv3(x); x2 = self.w3(x); x = x1 + x2
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

    def rollout(self, x0, grid, steps):
        preds = []
        x = x0.clone()
        for _ in range(steps):
            p = self.forward(x, grid)
            preds.append(p)
            x = torch.cat([x[:, :, :, 1:], p.unsqueeze(-1)], dim=-1)
        return torch.stack(preds, dim=1)


# ============================================================
# Data loading
# ============================================================

def load_data(mat_path, T_in=10, T_out=10, n_train=1000, n_test=200, target_res=64):
    try:
        import h5py
        with h5py.File(mat_path, 'r') as f:
            key = 'u' if 'u' in f.keys() else list(f.keys())[0]
            arr = np.array(f[key], dtype=np.float32)
    except Exception:
        import scipy.io
        data = scipy.io.loadmat(mat_path)
        keys = [k for k in data.keys() if not k.startswith('_')]
        key = 'u' if 'u' in keys else keys[0]
        arr = data[key].astype(np.float32)

    if arr.ndim == 4:
        if arr.shape[0] < arr.shape[-1]:
            u = torch.from_numpy(arr).permute(3, 1, 2, 0)
        else:
            u = torch.from_numpy(arr)

    N, H, W, T = u.shape
    if H != target_res or W != target_res:
        u = u.permute(0, 3, 1, 2).reshape(-1, 1, H, W)
        u = F.interpolate(u, size=(target_res, target_res),
                          mode='bilinear', align_corners=False)
        u = u.reshape(N, T, target_res, target_res).permute(0, 2, 3, 1)
        H, W = target_res, target_res

    gx = torch.linspace(0, 1, H)
    gy = torch.linspace(0, 1, W)
    grid = torch.stack(torch.meshgrid(gx, gy, indexing='ij'), dim=-1)

    x_test = u[n_train:n_train+n_test, :, :, :T_in]
    y_test = u[n_train:n_train+n_test, :, :, T_in:T_in+T_out]
    return x_test, y_test, grid, H, W


def paper_rel_l2(pred, target):
    """Per-sample relative L2 over full spatiotemporal volume."""
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = target.reshape(B, -1)
    per_sample = (p - t).norm(dim=1) / (t.norm(dim=1) + 1e-8)
    return per_sample.mean().item(), per_sample.std().item()


# ============================================================
# Long-horizon rollout: per-step error (Fig. 5a) and decoded enstrophy (Fig. 5b)
# ============================================================

@torch.no_grad()
def analysis_long_horizon(ckpt_koopman_1e5, ckpt_fno_1e5, data_1e5,
                          device, save_dir, args):
    print("\n" + "="*70)
    print("Long-horizon rollout: error vs time")
    print("  Koopman: single expm(L·t), no autoregression")
    print("  FNO: step-by-step autoregression, error compounds")
    print("="*70)

    # Load ν=1e-5 data — we have T=20, T_in=10, so 10 steps of ground truth
    x_test, y_test, grid, H, W = load_data(data_1e5, T_in=10, T_out=10)
    x_test, y_test = x_test.to(device), y_test.to(device)
    grid = grid.to(device)
    n_test = x_test.shape[0]
    target = y_test.permute(0, 3, 1, 2)  # (B, 10, H, W)

    # --- Koopman per-step error ---
    model_k = SpatialKoopmanLGN().to(device)
    model_k.load_state_dict(torch.load(ckpt_koopman_1e5, map_location=device, weights_only=False))
    model_k.eval()

    koopman_per_step = []
    for t_step in range(1, 11):
        # For each horizon, get prediction at that step only
        errs = []
        for i in range(0, n_test, 20):
            xb = x_test[i:i+20]; Bb = xb.shape[0]
            g = grid.unsqueeze(0).expand(Bb, -1, -1, -1)
            pred_all = model_k(xb, g, t_step)  # (B, t_step, H, W)
            pred_t = pred_all[:, -1]            # (B, H, W) — prediction at t_step
            tgt_t = target[i:i+20, t_step-1]   # (B, H, W) — ground truth at t_step
            B_ = pred_t.shape[0]
            per_sample = (pred_t.reshape(B_, -1) - tgt_t.reshape(B_, -1)).norm(dim=1) / \
                         (tgt_t.reshape(B_, -1).norm(dim=1) + 1e-8)
            errs.append(per_sample)
        errs = torch.cat(errs)
        koopman_per_step.append((errs.mean().item(), errs.std().item()))

    # --- FNO per-step error (autoregressive, error compounds) ---
    fno = FNO2d().to(device)
    fno.load_state_dict(torch.load(ckpt_fno_1e5, map_location=device, weights_only=False))
    fno.eval()

    fno_per_step = []
    # Run full 10-step rollout, then measure error at each step
    all_fno_preds = []
    for i in range(0, n_test, 20):
        xb = x_test[i:i+20]; Bb = xb.shape[0]
        g = grid.unsqueeze(0).expand(Bb, -1, -1, -1)
        pred = fno.rollout(xb, g, 10)  # (B, 10, H, W)
        all_fno_preds.append(pred.cpu())
    all_fno_preds = torch.cat(all_fno_preds).to(device)

    for t_step in range(10):
        pred_t = all_fno_preds[:, t_step]
        tgt_t = target[:, t_step]
        B_ = pred_t.shape[0]
        per_sample = (pred_t.reshape(B_, -1) - tgt_t.reshape(B_, -1)).norm(dim=1) / \
                     (tgt_t.reshape(B_, -1).norm(dim=1) + 1e-8)
        fno_per_step.append((per_sample.mean().item(), per_sample.std().item()))

    # Print
    print(f"\n  {'Step':<6} {'Koopman':>12} {'FNO':>12} {'Gap':>10}")
    print(f"  {'-'*42}")
    for t in range(10):
        km, _ = koopman_per_step[t]
        fm, _ = fno_per_step[t]
        gap = km / fm if fm > 0 else float('inf')
        print(f"  t={t+1:<4} {km:>12.4f} {fm:>12.4f} {gap:>10.2f}×")

    # --- Extended rollout beyond ground truth (stability check) ---
    print(f"\n  Extended rollout (no ground truth, checking for blowup)...")
    T_extended = 200

    # Koopman extended
    xb = x_test[:5]; Bb = 5
    g = grid.unsqueeze(0).expand(Bb, -1, -1, -1)
    pred_k_ext = model_k(xb, g, T_extended)
    k_max_val = pred_k_ext.abs().max().item()
    k_mean_energy = (pred_k_ext**2).mean(dim=(-2,-1))  # (5, T_extended)
    print(f"  Koopman T={T_extended}: max|ω|={k_max_val:.2f}, "
          f"stable={'YES' if k_max_val < 1e6 else 'BLOWUP'}")

    # FNO extended
    pred_f_ext = fno.rollout(xb, g, T_extended)
    f_max_val = pred_f_ext.abs().max().item()
    f_mean_energy = (pred_f_ext**2).mean(dim=(-2,-1))  # (5, T_extended)
    print(f"  FNO     T={T_extended}: max|ω|={f_max_val:.2f}, "
          f"stable={'YES' if f_max_val < 1e6 else 'BLOWUP'}")

    # --- Save CSVs ---
    import csv

    # CSV 1: Per-step error with ground truth (t=1..10)
    csv_path = os.path.join(save_dir, 'long_horizon_per_step.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'koopman_l2_mean', 'koopman_l2_std', 'fno_l2_mean', 'fno_l2_std'])
        for t in range(10):
            km, ks = koopman_per_step[t]
            fm, fs = fno_per_step[t]
            w.writerow([t+1, km, ks, fm, fs])
    print(f"  Saved: {csv_path}")

    # CSV 2: Extended energy rollout (t=1..T_extended)
    k_energy_arr = k_mean_energy.mean(dim=0).cpu().numpy()
    f_energy_arr = f_mean_energy.mean(dim=0).cpu().numpy()
    csv_path = os.path.join(save_dir, 'long_horizon_energy.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'koopman_energy', 'fno_energy'])
        for t in range(T_extended):
            w.writerow([t+1, float(k_energy_arr[t]), float(f_energy_arr[t])])
    print(f"  Saved: {csv_path}")

    # CSV 3: Summary scalars
    csv_path = os.path.join(save_dir, 'long_horizon_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['T_extended', T_extended])
        w.writerow(['koopman_max_omega', k_max_val])
        w.writerow(['fno_max_omega', f_max_val])
        w.writerow(['koopman_stable', int(k_max_val < 1e6)])
        w.writerow(['fno_stable', int(f_max_val < 1e6)])
        w.writerow(['koopman_t1_l2', koopman_per_step[0][0]])
        w.writerow(['fno_t1_l2', fno_per_step[0][0]])
    print(f"  Saved: {csv_path}")

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Per-step error (with ground truth, t=1..10)
    ax = axes[0]
    steps = np.arange(1, 11)
    k_means = [x[0] for x in koopman_per_step]
    k_stds  = [x[1] for x in koopman_per_step]
    f_means = [x[0] for x in fno_per_step]
    f_stds  = [x[1] for x in fno_per_step]

    ax.errorbar(steps, k_means, yerr=k_stds, fmt='o-', color='#2196F3',
                linewidth=2, capsize=3, label='LGN-KM (single expm)')
    ax.errorbar(steps, f_means, yerr=f_stds, fmt='s-', color='#FF5722',
                linewidth=2, capsize=3, label='FNO-2D (autoregressive)')
    ax.set_xlabel('Rollout Step t', fontsize=13)
    ax.set_ylabel('Per-step Relative L2', fontsize=13)
    ax.set_title('Error vs Horizon (with ground truth)', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks(steps)

    # Panel B: Extended rollout energy (stability, t=1..200)
    ax = axes[1]
    t_ext = np.arange(1, T_extended + 1)
    # Average energy over 5 samples
    k_energy = k_mean_energy.mean(dim=0).cpu().numpy()
    f_energy = f_mean_energy.mean(dim=0).cpu().numpy()
    ax.semilogy(t_ext, k_energy, color='#2196F3', linewidth=2,
                label='LGN-KM', alpha=0.8)
    ax.semilogy(t_ext, f_energy, color='#FF5722', linewidth=2,
                label='FNO-2D', alpha=0.8)
    ax.set_xlabel('Rollout Step t', fontsize=13)
    ax.set_ylabel('Mean Energy ||ω||²', fontsize=13)
    ax.set_title(f'Long-Horizon Stability (T={T_extended})', fontsize=14)
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig_long_horizon.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_1e5', required=True, help='data/ns/NavierStokes_V1e-5_N1200_T20.mat')
    parser.add_argument('--ckpt_koopman_1e5', required=True)
    parser.add_argument('--ckpt_fno_1e5',     required=True)
    parser.add_argument('--save_dir', default='outputs/fig5ab')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    analysis_long_horizon(
        args.ckpt_koopman_1e5, args.ckpt_fno_1e5, args.data_1e5,
        device, args.save_dir, args)


if __name__ == '__main__':
    main()
