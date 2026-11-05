"""
Release: produced the channel-dimension sweep of Appendix A.2 (Fig. A.1). The r = 32 runs are the
canonical models (checkpoints/ns/lgnkm_nu1e{5,3}_s0.pt), not re-run inside the sweep.
    python navier_stokes/ablations/ablation_r.py --data_1e5 data/ns/NavierStokes_V1e-5_N1200_T20.mat \
        --data_1e3 data/ns/ns_V1e-3_N5000_T50.mat --r_values 4 8 16 64 --save_dir runs/r_sweep
Released checkpoints and logs: checkpoints/ns_ablations/r_sweep/r{r}_{1e5,1e3}/ (best.pt, results.json, curve.csv).

Original docstring:
Channel Dimension Ablation: r = {4, 8, 16, 32, 64}
=====================================================
Trains LGN-KM at each r for both viscosities, logs:
  - Test L2 vs epoch (convergence curves)
  - Best test L2 per r
  - Eigenspectrum R² per r (does Kolmogorov scaling emerge?)
  - Parameter count per r
  - 100% stability check per r

Usage (run each on a separate GPU):
    CUDA_VISIBLE_DEVICES=0 python ablation_r.py --data_1e5 NavierStokes_V1e-5_N1200_T20.mat --viscosity 1e5
    CUDA_VISIBLE_DEVICES=1 python ablation_r.py --data_1e3 ns_V1e-3_N5000_T50.mat --viscosity 1e3

Or run all sequentially on one GPU:
    CUDA_VISIBLE_DEVICES=0 python ablation_r.py \
        --data_1e5 NavierStokes_V1e-5_N1200_T20.mat \
        --data_1e3 ns_V1e-3_N5000_T50.mat \
        --viscosity both

After all runs complete, generate the figure:
    python ablation_r.py --plot_only --save_dir ./ablation_results
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
import argparse, os, csv, json
from timeit import default_timer
from scipy.stats import linregress

torch.manual_seed(0)
np.random.seed(0)

R_VALUES = [4, 8, 16, 32, 64]

# ============================================================
# Model (identical to koopman_nu1e5_raw.py / koopman_nu1e3_raw.py)
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
    def __init__(self, modes, width, r, T_in, H, W):
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

    def propagate(self, z0_ft, t, dt=1.0):
        L_all = self.get_generators()
        eL = matrix_exp(L_all * t * dt)
        z = z0_ft.permute(0, 2, 3, 1).unsqueeze(-1)
        eL_exp = eL.unsqueeze(0)
        z_t = (eL_exp.to(torch.cfloat) @ z).squeeze(-1)
        return z_t.permute(0, 3, 1, 2)

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
        
    def get_max_re_eigenvalue(self):
        L_all = self.get_generators()
        L_flat = L_all.reshape(-1, self.r, self.r)
        eigs = torch.linalg.eigvals(L_flat)
        return eigs.real.max().item()

    def get_eigenspectrum_r2(self):
        """Compute R² of dominant Re(λ) vs |k|² regression."""
        L_all = self.get_generators()  # (modes, modes, r, r)
        M = self.modes
        k2_vals = []
        dom_re_vals = []
        for i in range(M):
            for j in range(M):
                if i == 0 and j == 0:
                    continue  # skip DC
                k2 = i**2 + j**2
                L_k = L_all[i, j]
                eigs = torch.linalg.eigvals(L_k)
                # dominant = least negative real part
                dom_re = eigs.real.max().item()
                k2_vals.append(k2)
                dom_re_vals.append(dom_re)

        k2_arr = np.array(k2_vals)
        re_arr = np.array(dom_re_vals)
        slope, intercept, r_value, p_value, std_err = linregress(k2_arr, re_arr)
        return r_value**2

    def get_stability_stats(self):
        """Return (max_re, pct_stable)."""
        L_all = self.get_generators()
        L_flat = L_all.reshape(-1, self.r, self.r)
        eigs = torch.linalg.eigvals(L_flat)
        all_re = eigs.real
        max_re = all_re.max().item()
        pct_stable = (all_re <= 0).float().mean().item() * 100
        return max_re, pct_stable


# ============================================================
# Metric & data loading (identical to training scripts)
# ============================================================

def paper_rel_l2_loss(pred, target):
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = target.reshape(B, -1)
    return ((p - t).norm(dim=1) / (t.norm(dim=1) + 1e-8)).mean()


def load_data(mat_path, T_in=10, T_out=10, n_train=1000, n_test=200, target_res=64):
    print(f"Loading {mat_path} ...")
    try:
        import h5py
        with h5py.File(mat_path, 'r') as f:
            keys = list(f.keys())
            key = 'u' if 'u' in keys else keys[0]
            arr = np.array(f[key], dtype=np.float32)
            print(f"  Format: HDF5, key='{key}', raw shape={arr.shape}")
    except Exception:
        import scipy.io
        data = scipy.io.loadmat(mat_path)
        keys = [k for k in data.keys() if not k.startswith('_')]
        key = 'u' if 'u' in keys else keys[0]
        arr = data[key].astype(np.float32)
        print(f"  Format: v5, key='{key}', raw shape={arr.shape}")

    if arr.ndim == 4:
        if arr.shape[0] < arr.shape[-1]:
            u = torch.from_numpy(arr).permute(3, 1, 2, 0)
        else:
            u = torch.from_numpy(arr)
    else:
        raise ValueError(f"Unexpected shape: {arr.shape}")

    N, H, W, T = u.shape
    print(f"  Parsed: N={N} H={H} W={W} T={T}")

    if H != target_res or W != target_res:
        print(f"  Downsampling {H}x{W} -> {target_res}x{target_res}")
        u = u.permute(0, 3, 1, 2).reshape(-1, 1, H, W)
        u = F.interpolate(u, size=(target_res, target_res),
                          mode='bilinear', align_corners=False)
        u = u.reshape(N, T, target_res, target_res).permute(0, 2, 3, 1)
        H, W = target_res, target_res

    assert T >= T_in + T_out

    gx = torch.linspace(0, 1, H)
    gy = torch.linspace(0, 1, W)
    grid = torch.stack(torch.meshgrid(gx, gy, indexing='ij'), dim=-1)

    x_train = u[:n_train,               :, :, :T_in]
    y_train = u[:n_train,               :, :, T_in:T_in+T_out]
    x_test  = u[n_train:n_train+n_test, :, :, :T_in]
    y_test  = u[n_train:n_train+n_test, :, :, T_in:T_in+T_out]

    print(f"  Train: {x_train.shape} -> {y_train.shape}")
    return x_train, y_train, x_test, y_test, grid, H, W


# ============================================================
# Single ablation run
# ============================================================

def train_one(r, data_path, T_out, tag, save_dir, epochs=500, T_in=10,
              modes=12, width=32, batch=10, lr=1e-3, n_train=1000, n_test=200):
    """Train one model, return dict of results."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_train, y_train, x_test, y_test, grid, H, W = load_data(
        data_path, T_in, T_out, n_train, n_test)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test,  y_test  = x_test.to(device),  y_test.to(device)
    grid = grid.to(device)
    n_train_actual, n_test_actual = x_train.shape[0], x_test.shape[0]

    # Reset seed for each run so r is the only variable
    torch.manual_seed(0)
    np.random.seed(0)

    model = SpatialKoopmanLGN(
        modes=modes, width=width, r=r, T_in=T_in, H=H, W=W).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  r={r}  tag={tag}  params={n_params:,}  T_out={T_out}")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    run_dir = os.path.join(save_dir, f'r{r}_{tag}')
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, 'curve.csv')
    with open(csv_path, 'w', newline='') as cf:
        csv.writer(cf).writerow(['epoch', 'train_l2', 'test_l2', 'max_re_eig', 'stable', 'time_s'])

    best_test_l2 = float('inf')
    best_epoch = 0

    for ep in range(epochs):
        model.train()
        t0 = default_timer()
        train_l2 = 0.0

        perm = torch.randperm(n_train_actual)
        for i in range(0, n_train_actual, batch):
            idx = perm[i:i+batch]
            xb = x_train[idx]; yb = y_train[idx]; Bb = xb.shape[0]
            g  = grid.unsqueeze(0).expand(Bb, -1, -1, -1)

            pred = model(xb, g, T_out)
            loss = paper_rel_l2_loss(pred, yb.permute(0, 3, 1, 2))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_l2 += loss.item() * Bb

        scheduler.step()
        train_l2 /= n_train_actual

        model.eval()
        with torch.no_grad():
            test_l2 = 0.0
            for i in range(0, n_test_actual, batch):
                xb = x_test[i:i+batch]; yb = y_test[i:i+batch]
                Bb = xb.shape[0]
                g  = grid.unsqueeze(0).expand(Bb, -1, -1, -1)
                pred = model(xb, g, T_out)
                test_l2 += paper_rel_l2_loss(pred, yb.permute(0, 3, 1, 2)).item() * Bb
            test_l2 /= n_test_actual
            max_re = model.get_max_re_eigenvalue()
            stable = 1 if max_re <= 0 else 0

        t1 = default_timer()

        with open(csv_path, 'a', newline='') as cf:
            csv.writer(cf).writerow([ep+1, f'{train_l2:.6f}', f'{test_l2:.6f}',
                                     f'{max_re:.6f}', stable, f'{t1-t0:.2f}'])

        if (ep + 1) % 50 == 0:
            flag = " <-- BEST" if test_l2 < best_test_l2 else ""
            stab = "stable" if stable else "UNSTABLE"
            print(f"  [r={r} {tag} ep={ep+1:04d}] train={train_l2:.4f} test={test_l2:.4f} "
                  f"Re_max={max_re:.5f} {stab}{flag}")

        if test_l2 < best_test_l2:
            best_test_l2 = test_l2
            best_epoch = ep + 1
            torch.save(model.state_dict(), os.path.join(run_dir, 'best.pt'))

    # ---- Post-training analysis on best model ----
    model.load_state_dict(torch.load(os.path.join(run_dir, 'best.pt'), weights_only=True))
    model.eval()
    with torch.no_grad():
        r2 = model.get_eigenspectrum_r2()
        max_re, pct_stable = model.get_stability_stats()

    results = {
        'r': r,
        'tag': tag,
        'n_params': n_params,
        'best_test_l2': best_test_l2,
        'best_epoch': best_epoch,
        'eigenspectrum_r2': r2,
        'max_re_eigenvalue': max_re,
        'pct_stable': pct_stable,
    }

    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  r={r} {tag} DONE: L2={best_test_l2:.4f} R²={r2:.3f} "
          f"stable={pct_stable:.1f}% params={n_params:,}")

    return results


# ============================================================
# Plotting
# ============================================================

def generate_figure(save_dir):
    """Load all results and generate ablation figure."""
    print("\nGenerating ablation figure...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = {'1e5': '#2196F3', '1e3': '#FF5722'}
    labels = {'1e5': 'ν=1e-5', '1e3': 'ν=1e-3'}

    for tag in ['1e5', '1e3']:
        rs, l2s, r2s, params_list = [], [], [], []
        for r in R_VALUES:
            json_path = os.path.join(save_dir, f'r{r}_{tag}', 'results.json')
            if not os.path.exists(json_path):
                print(f"  WARNING: missing {json_path}, skipping")
                continue
            with open(json_path) as f:
                res = json.load(f)
            rs.append(res['r'])
            l2s.append(res['best_test_l2'])
            r2s.append(res['eigenspectrum_r2'])
            params_list.append(res['n_params'])

        if not rs:
            continue

        c = colors[tag]
        lab = labels[tag]

        # Panel A: Test L2 vs r
        axes[0].plot(rs, l2s, 'o-', color=c, linewidth=2, markersize=8, label=lab)
        for r_val, l2_val in zip(rs, l2s):
            axes[0].annotate(f'{l2_val:.3f}', (r_val, l2_val),
                           textcoords="offset points", xytext=(0, 10),
                           fontsize=8, ha='center', color=c)

        # Panel B: Eigenspectrum R² vs r
        axes[1].plot(rs, r2s, 's-', color=c, linewidth=2, markersize=8, label=lab)
        for r_val, r2_val in zip(rs, r2s):
            axes[1].annotate(f'{r2_val:.3f}', (r_val, r2_val),
                           textcoords="offset points", xytext=(0, 10),
                           fontsize=8, ha='center', color=c)

        # Panel C: Params vs r
        axes[2].plot(rs, [p/1000 for p in params_list], 'D-', color=c,
                    linewidth=2, markersize=8, label=lab)

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
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='R²=0.9')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Channel Dimension r', fontsize=13)
    axes[2].set_ylabel('Parameters (K)', fontsize=13)
    axes[2].set_title('Model Size vs r', fontsize=14, fontweight='bold')
    axes[2].set_xticks(R_VALUES)
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'fig_ablation_r.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {fig_path}")

    # Also generate summary table
    print(f"\n{'='*80}")
    print(f"{'r':>4}  {'tag':>4}  {'params':>10}  {'test_L2':>8}  {'R²':>6}  {'stable%':>8}")
    print(f"{'-'*80}")
    for tag in ['1e5', '1e3']:
        for r in R_VALUES:
            json_path = os.path.join(save_dir, f'r{r}_{tag}', 'results.json')
            if not os.path.exists(json_path):
                continue
            with open(json_path) as f:
                res = json.load(f)
            print(f"{res['r']:>4}  {tag:>4}  {res['n_params']:>10,}  "
                  f"{res['best_test_l2']:>8.4f}  {res['eigenspectrum_r2']:>6.3f}  "
                  f"{res['pct_stable']:>7.1f}%")
    print(f"{'='*80}")

    # Save CSV summary
    csv_path = os.path.join(save_dir, 'ablation_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['r', 'viscosity', 'params', 'best_test_l2',
                     'eigenspectrum_r2', 'pct_stable', 'max_re_eigenvalue'])
        for tag in ['1e5', '1e3']:
            for r in R_VALUES:
                json_path = os.path.join(save_dir, f'r{r}_{tag}', 'results.json')
                if not os.path.exists(json_path):
                    continue
                with open(json_path) as f:
                    res = json.load(f)
                w.writerow([res['r'], tag, res['n_params'], res['best_test_l2'],
                           res['eigenspectrum_r2'], res['pct_stable'],
                           res['max_re_eigenvalue']])
    print(f"  Saved: {csv_path}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_1e5',  default=None, help='Path to ν=1e-5 data')
    parser.add_argument('--data_1e3',  default=None, help='Path to ν=1e-3 data')
    parser.add_argument('--viscosity', default='both', choices=['1e5', '1e3', 'both'])
    parser.add_argument('--epochs',    type=int, default=500)
    parser.add_argument('--save_dir',  default='./ablation_results')
    parser.add_argument('--plot_only', action='store_true')
    parser.add_argument('--r_values',  nargs='+', type=int, default=None,
                        help='Override r values (e.g. --r_values 4 8 16)')
    args = parser.parse_args()

    if args.r_values:
        R_VALUES[:] = args.r_values

    os.makedirs(args.save_dir, exist_ok=True)

    if args.plot_only:
        generate_figure(args.save_dir)
        exit()

    all_results = []

    # ν = 1e-5: T_out=10
    if args.viscosity in ['1e5', 'both']:
        assert args.data_1e5, "Need --data_1e5 for ν=1e-5 runs"
        for r in R_VALUES:
            res = train_one(r, args.data_1e5, T_out=10, tag='1e5',
                           save_dir=args.save_dir, epochs=args.epochs)
            all_results.append(res)

    # ν = 1e-3: T_out=40
    if args.viscosity in ['1e3', 'both']:
        assert args.data_1e3, "Need --data_1e3 for ν=1e-3 runs"
        for r in R_VALUES:
            res = train_one(r, args.data_1e3, T_out=40, tag='1e3',
                           save_dir=args.save_dir, epochs=args.epochs)
            all_results.append(res)

    generate_figure(args.save_dir)
    print("\nAll ablation runs complete.")
