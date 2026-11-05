"""
Continuous-Time Prediction: Koopman-LGN vs FNO
=================================================
Koopman: expm(L·t) for ANY t — one matrix exponential, no autoregression
FNO:     can only produce t=1,2,3,... via autoregression

This script:
    1. Evaluates both models at integer steps t=1..10 (standard)
    2. Evaluates Koopman at sub-integer steps t=0.5, 1.5, 2.5, ...
    3. Uses linear interpolation of ground truth for sub-integer validation
    4. Shows Koopman can query t=0.1, 0.5, 0.9 between any two frames
    5. Evaluates at arbitrary non-uniform times (t=π, t=e, etc.)
    6. Cost comparison: Koopman same cost for any t, FNO cost scales linearly

The paper uses experiment 3 only (Fig. 5c: wall time vs horizon). Its timing
is unaffected by the frame-index fix below; the accuracy columns of the other
experiments were one frame late in the original and are corrected here.

Usage (from the repository root):
    python continuous_time/continuous_time.py \
        --data data/ns/NavierStokes_V1e-5_N1200_T20.mat \
        --ckpt_koopman checkpoints/ns/lgnkm_nu1e5_s0.pt \
        --ckpt_fno checkpoints/ns/fno_nu1e5.pt \
        --save_dir outputs/continuous_time
Shipped output: continuous_time/results/continuous_time_cost_scaling.csv (Fig. 5c).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import matrix_exp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse, os, time

torch.manual_seed(0)
np.random.seed(0)

# ============================================================
# Models (same as before)
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

    def propagate(self, z0_ft, t, dt=1.0):
        L_all = self.get_generators()
        eL = matrix_exp(L_all * t * dt)
        z = z0_ft.permute(0, 2, 3, 1).unsqueeze(-1)
        z_t = (eL.unsqueeze(0).to(torch.cfloat) @ z).squeeze(-1)
        return z_t.permute(0, 3, 1, 2)

    def predict_at_time(self, x, grid, t_float, dt=1.0):
        """
        Predict at ARBITRARY continuous time t_float.
        t_float can be 0.5, 1.7, pi, anything > 0.
        Single expm evaluation — same cost regardless of t.
        """
        z0    = self.encoder(x, grid)
        z0_ft = torch.fft.rfft2(z0)
        z0_ft_trunc = z0_ft[:, :, :self.modes, :self.modes]

        zt_ft_trunc = self.propagate(z0_ft_trunc, t_float, dt)
        zt_ft = z0_ft.clone()
        zt_ft[:, :, :self.modes, :self.modes] = zt_ft_trunc
        z_t = torch.fft.irfft2(zt_ft, s=(self.H, self.W))
        return self.decoder(z_t)  # (B, H, W)

    def forward(self, x, grid, T_out, dt=1.0):
        z0    = self.encoder(x, grid)
        z0_ft = torch.fft.rfft2(z0)
        z0_ft_trunc = z0_ft[:, :, :self.modes, :self.modes]
        preds = []
        for t in range(1, T_out + 1):
            zt_ft_trunc = self.propagate(z0_ft_trunc, t, dt)
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

def load_data_full(mat_path, T_in=10, n_train=1000, n_test=200, target_res=64):
    """Load full time series (all T steps) for interpolation ground truth."""
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

    # Return full test trajectories for interpolation
    x_test = u[n_train:n_train+n_test, :, :, :T_in]                # (n_test, H, W, T_in)
    u_test_full = u[n_train:n_train+n_test, :, :, :]               # (n_test, H, W, T_total)
    return x_test, u_test_full, grid, H, W, T


def interpolate_gt(u_full, T_in, t_float):
    """
    Linear interpolation of ground truth at fractional time t_float.
    t_float is relative to T_in (t=1 means one step after input window).
    u_full: (B, H, W, T_total)
    Returns: (B, H, W)
    """
    # Model step t is frame index T_in - 1 + t (training targets are
    # u[..., T_in:T_in+T_out] for steps 1..T_out). The original used T_in + t,
    # one frame late, which roughly doubled every accuracy number; fixed for release.
    abs_t = T_in - 1 + t_float  # absolute time index
    t_lo = int(np.floor(abs_t))
    t_hi = t_lo + 1
    frac = abs_t - t_lo

    B, H, W, T = u_full.shape
    if t_hi >= T:
        return u_full[:, :, :, -1]  # clamp to last available

    return (1 - frac) * u_full[:, :, :, t_lo] + frac * u_full[:, :, :, t_hi]


def per_sample_rel_l2(pred, target):
    """pred, target: (B, H, W)"""
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = target.reshape(B, -1)
    return ((p - t).norm(dim=1) / (t.norm(dim=1) + 1e-8))


# ============================================================
# Main analysis
# ============================================================

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--ckpt_koopman', required=True)
    parser.add_argument('--ckpt_fno', required=True)
    parser.add_argument('--save_dir', default='./figures_continuous')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    x_test, u_full, grid, H, W, T_total = load_data_full(args.data)
    x_test = x_test.to(device)
    u_full = u_full.to(device)
    grid = grid.to(device)
    n_test = x_test.shape[0]
    print(f"Data: n_test={n_test}, H={H}, W={W}, T_total={T_total}")

    # Load models
    model_k = SpatialKoopmanLGN().to(device)
    model_k.load_state_dict(torch.load(args.ckpt_koopman, map_location=device, weights_only=False))
    model_k.eval()

    model_f = FNO2d().to(device)
    model_f.load_state_dict(torch.load(args.ckpt_fno, map_location=device, weights_only=False))
    model_f.eval()

    # ============================================================
    # EXPERIMENT 1: Dense temporal sampling
    # Koopman at dt=0.1 resolution vs FNO at dt=1.0
    # ============================================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: Dense Temporal Sampling")
    print("  Koopman: query any t (dt=0.1)")
    print("  FNO: integer steps only (dt=1.0)")
    print("="*70)

    # Max valid prediction offset (T=20, T_in=10 → index 19 → step 10).
    # The original capped this at 9, a symptom of the off-by-one fixed above.
    T_out_max = T_total - 10
    # Dense times for Koopman
    t_dense = np.arange(0.1, T_out_max + 0.01, 0.1)  # 0.1 to 9.0
    # Integer times for both
    t_integer = np.arange(1, T_out_max + 1)  # 1..9

    # Use first 50 test samples for speed
    n_eval = 50
    xb = x_test[:n_eval]
    Bb = xb.shape[0]
    g = grid.unsqueeze(0).expand(Bb, -1, -1, -1)

    # Koopman dense evaluation
    koopman_dense_errors = []
    koopman_dense_times_wall = []
    print(f"\n  Koopman dense evaluation ({len(t_dense)} time points)...")
    for t in t_dense:
        t0 = time.time()
        pred = model_k.predict_at_time(xb, g, t)      # (B, H, W)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        wall = time.time() - t0

        gt = interpolate_gt(u_full[:n_eval], 10, t)    # (B, H, W)
        errs = per_sample_rel_l2(pred, gt)
        koopman_dense_errors.append((t, errs.mean().item(), errs.std().item()))
        koopman_dense_times_wall.append(wall)

    # FNO integer evaluation (autoregressive)
    fno_integer_errors = []
    fno_wall_times = []
    print(f"  FNO integer evaluation ({len(t_integer)} steps, autoregressive)...")
    for t_int in t_integer:
        t0 = time.time()
        pred_all = model_f.rollout(xb, g, t_int)       # (B, t_int, H, W)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        wall = time.time() - t0

        pred = pred_all[:, -1]                          # last step
        gt = u_full[:n_eval, :, :, 9 + t_int]          # exact integer GT (step t = frame 9+t; was 10+t)
        errs = per_sample_rel_l2(pred, gt)
        fno_integer_errors.append((t_int, errs.mean().item(), errs.std().item()))
        fno_wall_times.append(wall)

    # Koopman at integers (for fair comparison)
    koopman_integer_errors = []
    koopman_wall_times = []
    print(f"  Koopman integer evaluation...")
    for t_int in t_integer:
        t0 = time.time()
        pred = model_k.predict_at_time(xb, g, float(t_int))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        wall = time.time() - t0

        gt = u_full[:n_eval, :, :, 9 + t_int]          # step t = frame 9+t (was 10+t)
        errs = per_sample_rel_l2(pred, gt)
        koopman_integer_errors.append((t_int, errs.mean().item(), errs.std().item()))
        koopman_wall_times.append(wall)

    # Print comparison
    print(f"\n  {'t':>5} {'Koopman':>12} {'FNO':>12} {'K wall(ms)':>12} {'F wall(ms)':>12}")
    print(f"  {'-'*55}")
    for i, t_int in enumerate(t_integer):
        km = koopman_integer_errors[i][1]
        fm = fno_integer_errors[i][1]
        kw = koopman_wall_times[i] * 1000
        fw = fno_wall_times[i] * 1000
        print(f"  {t_int:>5} {km:>12.4f} {fm:>12.4f} {kw:>12.1f} {fw:>12.1f}")

    # ============================================================
    # EXPERIMENT 2: Arbitrary time queries
    # ============================================================
    print(f"\n" + "="*70)
    print("EXPERIMENT 2: Arbitrary Time Queries")
    print("  Times FNO literally cannot produce without retraining")
    print("="*70)

    special_times = [0.5, 1.5, np.pi, np.e, 4.5, 7.25, 8.5]
    print(f"\n  {'t':>8} {'Koopman L2':>12} {'Note'}")
    print(f"  {'-'*50}")
    for t in special_times:
        pred = model_k.predict_at_time(xb, g, t)
        gt = interpolate_gt(u_full[:n_eval], 10, t)
        errs = per_sample_rel_l2(pred, gt)
        note = ""
        if t == np.pi: note = "  t=π"
        elif t == np.e: note = "  t=e"
        elif t % 1 == 0.5: note = "  half-step"
        print(f"  {t:>8.4f} {errs.mean().item():>12.4f}{note}")

    # ============================================================
    # EXPERIMENT 3: Temporal cost scaling
    # ============================================================
    print(f"\n" + "="*70)
    print("EXPERIMENT 3: Computational Cost Scaling")
    print("  FNO: cost ∝ T (autoregressive)")
    print("  Koopman: cost = O(1) (single expm)")
    print("="*70)

    horizons = [1, 5, 10, 20, 50, 100, 200]
    koopman_costs = []
    fno_costs = []

    xb_small = x_test[:10]
    Bb_small = 10
    g_small = grid.unsqueeze(0).expand(Bb_small, -1, -1, -1)

    # Warm up
    _ = model_k.predict_at_time(xb_small, g_small, 1.0)
    _ = model_f.rollout(xb_small, g_small, 1)

    for T in horizons:
        # Koopman: single expm at t=T
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        for _ in range(5):  # average over 5 runs
            pred_k = model_k.predict_at_time(xb_small, g_small, float(T))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        koopman_costs.append((time.time() - t0) / 5 * 1000)

        # FNO: autoregressive T steps
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        for _ in range(5):
            pred_f = model_f.rollout(xb_small, g_small, T)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        fno_costs.append((time.time() - t0) / 5 * 1000)

    print(f"\n  {'T':>6} {'Koopman(ms)':>14} {'FNO(ms)':>14} {'Speedup':>10}")
    print(f"  {'-'*48}")
    for i, T in enumerate(horizons):
        speedup = fno_costs[i] / koopman_costs[i] if koopman_costs[i] > 0 else float('inf')
        print(f"  {T:>6} {koopman_costs[i]:>14.1f} {fno_costs[i]:>14.1f} {speedup:>10.1f}×")

    # ============================================================
    # SAVE CSVs
    # ============================================================
    import csv

    # CSV 1: Koopman dense errors (dt=0.1)
    csv_path = os.path.join(args.save_dir, 'continuous_time_koopman_dense.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'rel_l2_mean', 'rel_l2_std'])
        for t, m, s in koopman_dense_errors:
            w.writerow([t, m, s])
    print(f"  Saved: {csv_path}")

    # CSV 2: Integer step comparison (Koopman vs FNO)
    csv_path = os.path.join(args.save_dir, 'continuous_time_integer_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'koopman_l2_mean', 'koopman_l2_std',
                     'fno_l2_mean', 'fno_l2_std',
                     'koopman_wall_ms', 'fno_wall_ms'])
        for i, t_int in enumerate(t_integer):
            w.writerow([int(t_int),
                        koopman_integer_errors[i][1], koopman_integer_errors[i][2],
                        fno_integer_errors[i][1], fno_integer_errors[i][2],
                        koopman_wall_times[i] * 1000, fno_wall_times[i] * 1000])
    print(f"  Saved: {csv_path}")

    # CSV 3: Special/arbitrary time queries
    csv_path = os.path.join(args.save_dir, 'continuous_time_special_queries.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'label', 'koopman_l2_mean'])
        for t in special_times:
            pred = model_k.predict_at_time(xb, g, t)
            gt = interpolate_gt(u_full[:n_eval], 10, t)
            errs = per_sample_rel_l2(pred, gt)
            label = ''
            if abs(t - np.pi) < 1e-6: label = 'pi'
            elif abs(t - np.e) < 1e-6: label = 'e'
            elif t % 1 == 0.5: label = 'half-step'
            w.writerow([t, label, errs.mean().item()])
    print(f"  Saved: {csv_path}")

    # CSV 4: Cost scaling
    csv_path = os.path.join(args.save_dir, 'continuous_time_cost_scaling.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['horizon_T', 'koopman_ms', 'fno_ms', 'speedup'])
        for i, T in enumerate(horizons):
            sp = fno_costs[i] / koopman_costs[i] if koopman_costs[i] > 0 else float('inf')
            w.writerow([T, koopman_costs[i], fno_costs[i], sp])
    print(f"  Saved: {csv_path}")

    # ============================================================
    # FIGURE
    # ============================================================
    print("\n  Generating figures...")

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, hspace=0.35, wspace=0.35)

    c_k = '#2196F3'
    c_f = '#FF5722'
    c_dense = '#9C27B0'

    # --- Panel A: Error vs time (dense Koopman + integer FNO) ---
    ax = fig.add_subplot(gs[0, 0])
    # Koopman dense
    t_d = [x[0] for x in koopman_dense_errors]
    e_d = [x[1] for x in koopman_dense_errors]
    ax.plot(t_d, e_d, '-', color=c_dense, linewidth=1.5, alpha=0.7,
            label='Koopman (dt=0.1)')
    # Koopman integer
    t_ki = [x[0] for x in koopman_integer_errors]
    e_ki = [x[1] for x in koopman_integer_errors]
    ax.plot(t_ki, e_ki, 'o', color=c_k, markersize=8, zorder=5,
            label='Koopman (integer)')
    # FNO integer
    t_fi = [x[0] for x in fno_integer_errors]
    e_fi = [x[1] for x in fno_integer_errors]
    ax.plot(t_fi, e_fi, 's', color=c_f, markersize=8, zorder=5,
            label='FNO (integer only)')
    # Shade the region FNO can't access
    ax.axhspan(-0.01, max(e_d)*1.1, xmin=0, xmax=1, alpha=0.03, color='gray')
    ax.set_xlabel('Time t (after input window)', fontsize=13)
    ax.set_ylabel('Relative L2', fontsize=13)
    ax.set_title('Continuous-Time Prediction\nKoopman queries any t, FNO integers only', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim([0, T_out_max + 0.5])

    # --- Panel B: Sub-integer predictions (visual) ---
    ax = fig.add_subplot(gs[0, 1])
    sample_idx = 0
    sub_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    # Plot ground truth and Koopman predictions at sub-integer times
    for i, t in enumerate(sub_times):
        pred = model_k.predict_at_time(xb[:1], g[:1], t)
        gt = interpolate_gt(u_full[:1], 10, t)
        err = per_sample_rel_l2(pred, gt).item()
        # Just show error evolution
        marker = 'o' if t % 1 == 0 else 'D'
        color = c_k if t % 1 == 0 else c_dense
        ax.scatter(t, err, s=100, c=color, marker=marker, zorder=5, edgecolors='k')
        ax.annotate(f't={t}', (t, err), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9)

    ax.set_xlabel('Time t', fontsize=13)
    ax.set_ylabel('Relative L2 (sample 0)', fontsize=13)
    ax.set_title('Sub-Integer Predictions\n○ = integer (training grid), ◇ = fractional', fontsize=14)

    # --- Panel C: Cost scaling ---
    ax = fig.add_subplot(gs[0, 2])
    ax.loglog(horizons, koopman_costs, 'o-', color=c_k, linewidth=2, markersize=8,
              label='Koopman-LGN')
    ax.loglog(horizons, fno_costs, 's-', color=c_f, linewidth=2, markersize=8,
              label='FNO-2D')
    ax.set_xlabel('Prediction Horizon T', fontsize=13)
    ax.set_ylabel('Wall Time (ms)', fontsize=13)
    ax.set_title('Computational Cost Scaling\nKoopman: O(1), FNO: O(T)', fontsize=14)
    ax.legend(fontsize=11)
    # Annotate speedup at T=200
    if len(horizons) > 0:
        speedup_200 = fno_costs[-1] / koopman_costs[-1]
        ax.annotate(f'{speedup_200:.0f}× speedup\nat T={horizons[-1]}',
                    xy=(horizons[-1], koopman_costs[-1]),
                    xytext=(horizons[-1]*0.15, koopman_costs[-1]*0.5),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=11, color='green', fontweight='bold')

    # --- Panel D: Visual comparison at t=3.5 (sub-integer) ---
    t_show = 3.5
    pred_k = model_k.predict_at_time(xb[:1], g[:1], t_show)
    gt_show = interpolate_gt(u_full[:1], 10, t_show)
    err_show = per_sample_rel_l2(pred_k, gt_show).item()

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(gt_show[0].cpu().numpy(), cmap='RdBu_r', aspect='equal')
    ax.set_title(f'Ground Truth at t={t_show}\n(interpolated)', fontsize=13)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(pred_k[0].cpu().numpy(), cmap='RdBu_r', aspect='equal')
    ax.set_title(f'Koopman at t={t_show}\n(single expm, L2={err_show:.3f})', fontsize=13)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 2])
    diff = (pred_k[0] - gt_show[0]).cpu().numpy()
    vmax = max(abs(diff.min()), abs(diff.max()))
    im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_title(f'Error at t={t_show}\n(FNO cannot produce this)', fontsize=13)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.savefig(os.path.join(args.save_dir, 'fig_continuous_time.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {args.save_dir}/fig_continuous_time.png")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print(f"CONTINUOUS-TIME RESULTS")
    print(f"{'='*70}")
    print(f"""
  Koopman-LGN evaluates at ANY continuous time via expm(L·t).
  FNO can only produce integer steps via autoregression.

  At integer steps (fair comparison):
    Koopman t=1: {koopman_integer_errors[0][1]:.4f}   FNO t=1: {fno_integer_errors[0][1]:.4f}
    Koopman t=5: {koopman_integer_errors[4][1]:.4f}   FNO t=5: {fno_integer_errors[4][1]:.4f}
    Koopman t=9: {koopman_integer_errors[-1][1]:.4f}  FNO t=9: {fno_integer_errors[-1][1]:.4f}

  Koopman at sub-integer times (FNO impossible):
    t=0.5: {[x[1] for x in koopman_dense_errors if abs(x[0]-0.5)<0.01][0]:.4f}
    t=π:   evaluated (see figure)
    t=e:   evaluated (see figure)

  Cost scaling at T=200:
    Koopman: {koopman_costs[-1]:.1f} ms
    FNO:     {fno_costs[-1]:.1f} ms
    Speedup: {fno_costs[-1]/koopman_costs[-1]:.0f}×
""")


if __name__ == '__main__':
    main()
