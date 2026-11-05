"""
Cross-viscosity transfer experiment for LGN-KM
================================================
Pretrain: ν=1e-5 (checkpoint)
Target:   ν=1e-3 (fine-tune with varying data budgets)

Three conditions at each data budget [50, 100, 200, 500, 1000]:
  1. scratch       — random init, train on ν=1e-3
  2. transfer_all  — load ν=1e-5 checkpoint, fine-tune everything
  3. transfer_freezeS — load checkpoint, freeze S (P matrix), fine-tune rest

Paper: Section 4.2, Fig. 6, Appendix A.3 / Table A.1. The released runs are
seed 0 only (checkpoints/transfer/, curves in transfer/results/).

Learning rate: the two transfer conditions fine-tune at 0.3 x --lr
(3e-4 for the default 1e-3); scratch trains at --lr. This is the setting that
produced the paper's numbers. The log header prints the learning rate actually used.
Same data pool (first N of trajectories 0:1000) and test set (1000:1200) as the
main nu=1e-3 model; checkpoint = lowest test L2 over 500 epochs.

Usage (from the repo root):
    # One condition (the paper ran the 15 (condition, N) pairs in parallel):
    python transfer/transfer_experiment.py \
        --pretrained checkpoints/ns/lgnkm_nu1e5_s0.pt \
        --data data/ns/ns_V1e-3_N5000_T50.mat \
        --condition transfer_freezeS --n_train 100 --save_dir transfer/runs

    # All conditions and budgets sequentially:
    python transfer/transfer_experiment.py \
        --pretrained checkpoints/ns/lgnkm_nu1e5_s0.pt \
        --data data/ns/ns_V1e-3_N5000_T50.mat --save_dir transfer/runs

    # Fig. 6(b,c) numbers from the curves:
    python transfer/fig6_metrics.py --results_dir transfer/results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import matrix_exp
import numpy as np
import argparse, os, csv, json, copy
from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)


# ============================================================
# Model (identical to koopman_nu1e5_raw.py)
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


def paper_rel_l2_loss(pred, target):
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = target.reshape(B, -1)
    return ((p - t).norm(dim=1) / (t.norm(dim=1) + 1e-8)).mean()


# ============================================================
# Data loading (from koopman_nu1e3_raw.py, NO normalization)
# ============================================================

def load_data(mat_path, T_in=10, T_out=40, n_train=1000, n_test=200, target_res=64):
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

    # Return full dataset — caller slices by n_train
    x_all = u[:, :, :, :T_in]
    y_all = u[:, :, :, T_in:T_in+T_out]

    # Fixed test set: always the same 200 samples from the end
    x_test = x_all[n_train:n_train+n_test]
    y_test = y_all[n_train:n_train+n_test]
    # Train pool: first n_train samples (caller subsets further)
    x_train_pool = x_all[:n_train]
    y_train_pool = y_all[:n_train]

    print(f"  Train pool: {x_train_pool.shape}, Test: {x_test.shape}")
    return x_train_pool, y_train_pool, x_test, y_test, grid, H, W


# ============================================================
# Single training run
# ============================================================

def run_one(condition, n_sub, pretrained_path, data_path, save_dir,
            epochs=500, lr=1e-3, batch=10, T_in=10, T_out=40,
            r=32, modes=12, width=32, seed=0):
    """
    condition: 'scratch' | 'transfer_all' | 'transfer_freezeS'
    n_sub:     number of training samples to use
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tag = f"{condition}_N{n_sub}_s{seed}"
    # header shows the lr actually used (originally printed the nominal --lr)
    print(f"\n{'='*60}")
    print(f"  {tag}  |  epochs={epochs}  lr={lr * 0.3 if condition.startswith('transfer') else lr}")
    print(f"{'='*60}")

    # --- Load data ---
    x_pool, y_pool, x_test, y_test, grid, H, W = load_data(
        data_path, T_in, T_out, n_train=1000, n_test=200)

    # Subset training data
    x_train = x_pool[:n_sub].to(device)
    y_train = y_pool[:n_sub].to(device)
    x_test  = x_test.to(device)
    y_test  = y_test.to(device)
    grid    = grid.to(device)
    n_test  = x_test.shape[0]
    print(f"  Using {n_sub} training samples, {n_test} test samples")

    # --- Build model ---
    model = SpatialKoopmanLGN(modes=modes, width=width, r=r,
                               T_in=T_in, H=H, W=W).to(device)

    # --- Initialize weights ---
    if condition in ('transfer_all', 'transfer_freezeS'):
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        print(f"  Loaded pretrained: {pretrained_path}")

        if condition == 'transfer_freezeS':
            model.P.requires_grad = False
            print(f"  Frozen: P (S matrix) — {model.P.numel()} params")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable:,} trainable / {total:,} total")

    # --- Optimizer ---
    # Lower LR for transfer to avoid catastrophic forgetting
    actual_lr = lr * 0.3 if condition.startswith('transfer') else lr
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=actual_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Logging ---
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f'curve_{tag}.csv')
    with open(csv_path, 'w', newline='') as cf:
        csv.writer(cf).writerow(['epoch', 'train_l2', 'test_l2',
                                  'max_re_eig', 'stable', 'time_s'])

    best_test = float('inf')

    for ep in range(epochs):
        model.train()
        t0 = default_timer()
        train_l2 = 0.0

        perm = torch.randperm(n_sub)
        for i in range(0, n_sub, batch):
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
        train_l2 /= n_sub

        model.eval()
        with torch.no_grad():
            test_l2 = 0.0
            for i in range(0, n_test, batch):
                xb = x_test[i:i+batch]; yb = y_test[i:i+batch]
                Bb = xb.shape[0]
                g  = grid.unsqueeze(0).expand(Bb, -1, -1, -1)
                pred = model(xb, g, T_out)
                test_l2 += paper_rel_l2_loss(pred, yb.permute(0, 3, 1, 2)).item() * Bb
            test_l2 /= n_test
            max_re = model.get_max_re_eigenvalue()
            stable = 1 if max_re <= 0 else 0

        t1 = default_timer()

        with open(csv_path, 'a', newline='') as cf:
            csv.writer(cf).writerow([ep+1, f'{train_l2:.6f}', f'{test_l2:.6f}',
                                     f'{max_re:.6f}', stable, f'{t1-t0:.2f}'])

        if (ep + 1) % 50 == 0:
            flag = " <-- BEST" if test_l2 < best_test else ""
            print(f"  [{tag}][{ep+1:04d}] train={train_l2:.4f} test={test_l2:.4f} "
                  f"Re_max={max_re:.5f}{flag}")

        if test_l2 < best_test:
            best_test = test_l2
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f'best_{tag}.pt'))

    print(f"  DONE [{tag}]  best_test={best_test:.4f}")
    return tag, best_test


# ============================================================
# Main — run all conditions or a single one
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', required=True,
                        help='Path to best_koopman_nu1e5_raw.pt')
    parser.add_argument('--data', required=True,
                        help='Path to ns_V1e-3_N5000_T50.mat')
    parser.add_argument('--save_dir', default='./transfer_results')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=10)

    # For single-run mode (parallel jobs):
    parser.add_argument('--condition', default=None,
                        choices=['scratch', 'transfer_all', 'transfer_freezeS'])
    parser.add_argument('--n_train', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    data_budgets = [50, 100, 200, 500, 1000]
    conditions   = ['scratch', 'transfer_all', 'transfer_freezeS']

    results = {}

    if args.condition and args.n_train:
        # Single run mode
        tag, best = run_one(
            args.condition, args.n_train, args.pretrained, args.data,
            args.save_dir, args.epochs, args.lr, args.batch, seed=args.seed)
        results[tag] = best
    else:
        # Full sweep
        for n_sub in data_budgets:
            for cond in conditions:
                tag, best = run_one(
                    cond, n_sub, args.pretrained, args.data,
                    args.save_dir, args.epochs, args.lr, args.batch)
                results[tag] = best

    # --- Summary ---
    summary_path = os.path.join(args.save_dir, 'transfer_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    print(f"\n{'='*70}")
    print(f"  TRANSFER EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    for tag, best in sorted(results.items()):
        print(f"  {tag:40s}  test_l2 = {best:.4f}")
