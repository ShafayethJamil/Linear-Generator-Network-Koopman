"""
LGN-KM with a DCT-II (cosine) lift, no-flux FHN (Appendix A.7, Table A.4)
==========================================================================
Same S-D generator, encoder/decoder shape, protocol and metric as the periodic
FHN runs. Only change: every rfft2/irfft2 is replaced by an orthonormal DCT-II,
the basis that satisfies homogeneous Neumann BCs, so the whole model is real.
D_k is indexed by cosine wavenumbers k_m = pi m / L.

Usage (paper models: 2 diffusivities x seeds 0, 1, 2):
    python fitzhugh_nagumo/train_fhn_neumann.py --data data/fhn/rd_neumann_Du0.01_N1200_T30.mat \
        --tag fhn_dct_Du0.01_s0 --seed 0
    python fitzhugh_nagumo/train_fhn_neumann.py --data <same .mat> --tag <same tag> --analyze
        -> eigenspectrum_<tag>.csv and dct_summary_<tag>.csv (max Re, mean decay vs |k|^2)
Note: the mean decay vs |k|^2 is linear by construction (tr S = 0, so the mean of
-Re(lambda) over a wavevector equals the mean of diag D_k); its R^2 = 1 is an
identity, not a fit result.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import matrix_exp
import numpy as np
import argparse, os, csv
from timeit import default_timer


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ============================================================
# Orthonormal DCT-II matrix (res x res):
#   C[m, j] = s_m * cos(pi*(j+0.5)*m/res),  s_0=sqrt(1/res), s_m=sqrt(2/res)
# 2D DCT of x (..., H, W):  C @ x @ C^T ;  inverse:  C^T @ x_hat @ C
# ============================================================

def dct_matrix(res):
    j = torch.arange(res, dtype=torch.float64)
    m = torch.arange(res, dtype=torch.float64)
    C = torch.cos(np.pi * (j.unsqueeze(0) + 0.5) * m.unsqueeze(1) / res)
    C[0, :] *= np.sqrt(1.0 / res)
    C[1:, :] *= np.sqrt(2.0 / res)
    return C.to(torch.float32)


def dct2(x, C):
    # x: (..., H, W)
    return torch.matmul(C, torch.matmul(x, C.transpose(0, 1)))


def idct2(x_hat, C):
    return torch.matmul(C.transpose(0, 1), torch.matmul(x_hat, C))


# ============================================================
# Cosine-spectral Conv 2D (DCT analogue of SpectralConv2d; real weights)
# ============================================================

class CosineConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / (in_ch * out_ch)
        self.w = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2))

    def forward(self, x, C):
        B, Ch, H, W = x.shape
        x_hat = dct2(x, C)
        out = torch.zeros(B, self.w.shape[1], H, W, device=x.device, dtype=x.dtype)
        out[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_hat[:, :, :self.modes1, :self.modes2], self.w)
        return idct2(out, C)


# ============================================================
# Encoder / Decoder (same shape as paper; convs are cosine-spectral)
# ============================================================

class SpatialEncoderDCT(nn.Module):
    def __init__(self, modes, width, r, T_in):
        super().__init__()
        self.fc0   = nn.Linear(T_in + 2, width)
        self.conv0 = CosineConv2d(width, width, modes, modes)
        self.conv1 = CosineConv2d(width, width, modes, modes)
        self.w0    = nn.Conv2d(width, width, 1)
        self.w1    = nn.Conv2d(width, width, 1)
        self.proj  = nn.Conv2d(width, r, 1)

    def forward(self, x, grid, C):
        x = torch.cat([x, grid], dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        x = F.gelu(self.conv0(x, C) + self.w0(x))
        x = F.gelu(self.conv1(x, C) + self.w1(x))
        return self.proj(x)


class SpatialDecoder(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(r, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))

    def forward(self, z):
        return self.net(z).squeeze(1)


# ============================================================
# LGN-KM with DCT lift. Generator UNCHANGED: L_k = S - D_k,
# S = (P - P^T)/2 shared, D_k = softplus(d) + softplus(alpha)*|k|^2_norm.
# k now indexes COSINE modes: k2_norm = (m1^2 + m2^2)/modes^2.
# Everything is real -> matrix_exp and propagation in real arithmetic.
# ============================================================

class SpatialKoopmanLGN_DCT(nn.Module):
    def __init__(self, modes, width, r, T_in, H, W):
        super().__init__()
        self.modes, self.r, self.H, self.W = modes, r, H, W
        self.encoder = SpatialEncoderDCT(modes, width, r, T_in)
        self.decoder = SpatialDecoder(r)
        self.P     = nn.Parameter(torch.randn(r, r) * 0.01)
        self.d     = nn.Parameter(torch.full((r,), -2.0))
        self.alpha = nn.Parameter(torch.full((r,), -3.0))

        m1 = torch.arange(modes, dtype=torch.float32)
        m2 = torch.arange(modes, dtype=torch.float32)
        mm1, mm2 = torch.meshgrid(m1, m2, indexing='ij')
        k2_norm = (mm1**2 + mm2**2) / (modes**2)
        self.register_buffer('k2_norm', k2_norm)
        self.register_buffer('C', dct_matrix(H))

    def get_generators(self):
        S  = (self.P - self.P.T) / 2.0
        d0 = F.softplus(self.d)
        a0 = F.softplus(self.alpha)
        D_diag = d0.unsqueeze(0).unsqueeze(0) + \
                 a0.unsqueeze(0).unsqueeze(0) * self.k2_norm.unsqueeze(-1)
        S_exp = S.unsqueeze(0).unsqueeze(0).expand(self.modes, self.modes, -1, -1).clone()
        D_mat = torch.diag_embed(D_diag)
        return S_exp - D_mat

    def propagate(self, z0_hat, t, dt=1.0):
        L_all = self.get_generators()
        eL = matrix_exp(L_all * t * dt)                    # (m, m, r, r) REAL
        z = z0_hat.permute(0, 2, 3, 1).unsqueeze(-1)       # (B, m, m, r, 1)
        z_t = (eL.unsqueeze(0) @ z).squeeze(-1)
        return z_t.permute(0, 3, 1, 2)

    def forward(self, x, grid, T_out, dt=1.0):
        z0     = self.encoder(x, grid, self.C)             # (B, r, H, W)
        z0_hat = dct2(z0, self.C)
        z0_hat_trunc = z0_hat[:, :, :self.modes, :self.modes]

        preds = []
        for t in range(1, T_out + 1):
            zt_hat_trunc = self.propagate(z0_hat_trunc, t, dt)
            zt_hat = z0_hat.clone()
            zt_hat[:, :, :self.modes, :self.modes] = zt_hat_trunc
            z_t = idct2(zt_hat, self.C)
            preds.append(self.decoder(z_t))
        return torch.stack(preds, dim=1)

    def get_max_re_eigenvalue(self):
        L_all = self.get_generators()
        L_flat = L_all.reshape(-1, self.r, self.r)
        eigs = torch.linalg.eigvals(L_flat)
        return eigs.real.max().item()


# ============================================================
# Paper metric / data loading (identical to koopman_FHN.py)
# ============================================================

def paper_rel_l2_loss(pred, target):
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = target.reshape(B, -1)
    return ((p - t).norm(dim=1) / (t.norm(dim=1) + 1e-8)).mean()


def load_data(mat_path, T_in=10, T_out=20, n_train=1000, n_test=200, target_res=64):
    print(f"Loading {mat_path} ...")
    import h5py
    with h5py.File(mat_path, 'r') as f:
        keys = list(f.keys())
        key = 'u' if 'u' in keys else keys[0]
        arr = np.array(f[key], dtype=np.float32)
        print(f"  Format: HDF5, key='{key}', raw shape={arr.shape}")

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

    assert T >= T_in + T_out, f"T={T} < T_in+T_out={T_in+T_out}"

    gx = torch.linspace(0, 1, H)
    gy = torch.linspace(0, 1, W)
    grid = torch.stack(torch.meshgrid(gx, gy, indexing='ij'), dim=-1)

    x_train = u[:n_train,               :, :, :T_in]
    y_train = u[:n_train,               :, :, T_in:T_in+T_out]
    x_test  = u[n_train:n_train+n_test, :, :, :T_in]
    y_test  = u[n_train:n_train+n_test, :, :, T_in:T_in+T_out]

    print(f"  Train: {x_train.shape} -> {y_train.shape}")
    print(f"  Data stats: mean={u.mean():.4f} std={u.std():.4f} (NO normalization)")
    return x_train, y_train, x_test, y_test, grid, H, W


# ============================================================
# Post-hoc spectral readout (trace estimator; linear fit vs |k|^2)
# ============================================================

def analyze(model, tag, save_dir):
    model.eval()
    modes, r = model.modes, model.r
    with torch.no_grad():
        L_all = model.get_generators().cpu()
        k2n = model.k2_norm.cpu().numpy()

    rows, shells = [], {}
    n_total, n_stable, max_re = 0, 0, -np.inf
    for i in range(modes):
        for j in range(modes):
            eigs = torch.linalg.eigvals(L_all[i, j]).numpy()
            n_total += eigs.size
            n_stable += int((eigs.real <= 1e-9).sum())
            max_re = max(max_re, float(eigs.real.max()))
            for e in eigs:
                rows.append([e.real, e.imag, k2n[i, j]])
            shells.setdefault(round(float(k2n[i, j]), 8), []).append(eigs.real.mean())

    eig_path = os.path.join(save_dir, f'eigenspectrum_{tag}.csv')
    with open(eig_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['re', 'im', 'k2']); w.writerows(rows)

    ks = np.array(sorted(shells.keys()))
    mean_decay = np.array([-np.mean(shells[k]) for k in ks])  # = mean diag D (tr S = 0)
    A = np.vstack([np.ones_like(ks), ks]).T
    (c, slope), *_ = np.linalg.lstsq(A, mean_decay, rcond=None)
    yfit = A @ np.array([c, slope])
    r2 = 1 - ((mean_decay-yfit)**2).sum() / ((mean_decay-mean_decay.mean())**2).sum()

    summ = os.path.join(save_dir, f'dct_summary_{tag}.csv')
    with open(summ, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['slope_meandecay_vs_k2', f'{slope:.6f}'])
        w.writerow(['intercept', f'{c:.6f}'])
        w.writerow(['r2_linear_k2', f'{r2:.5f}'])
        w.writerow(['max_re_lambda', f'{max_re:.6f}'])
        w.writerow(['pct_stable', f'{100.0*n_stable/n_total:.2f}'])

    print(f"\n{'='*62}\nDCT-LGN ANALYSIS [{tag}] (cosine basis, no-flux BCs)")
    print(f"  mean-decay vs |k|^2:  slope = {slope:.5f}, R2 = {r2:.4f}")
    print(f"  max Re(lambda) = {max_re:.6f} | stable = {100.0*n_stable/n_total:.1f}%")
    print(f"  Train both Du values, then compare slope ratio to Du ratio (10x).")
    print(f"  wrote {eig_path}\n  wrote {summ}\n{'='*62}")


# ============================================================
# Training (identical loop)
# ============================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    x_train, y_train, x_test, y_test, grid, H, W = load_data(
        args.data, args.T_in, args.T_out, args.n_train, args.n_test)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test,  y_test  = x_test.to(device),  y_test.to(device)
    grid = grid.to(device)
    n_train, n_test = x_train.shape[0], x_test.shape[0]

    model = SpatialKoopmanLGN_DCT(
        modes=args.modes, width=args.width, r=args.r,
        T_in=args.T_in, H=H, W=W).to(device)

    ckpt_path = os.path.join(args.save_dir, f'best_koopman_{args.tag}.pt')

    if args.analyze:
        print(f"Loading {ckpt_path} for analysis ...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        analyze(model, args.tag, args.save_dir)
        return

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DCT-LGN params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, f'curve_koopman_{args.tag}.csv')
    with open(csv_path, 'w', newline='') as cf:
        csv.writer(cf).writerow(['epoch', 'train_l2', 'test_l2', 'max_re_eig', 'stable', 'time_s'])

    best_test_l2 = float('inf')
    print(f"\n[{args.tag}] DCT-LGN (Neumann/cosine)  T_in={args.T_in} T_out={args.T_out} "
          f"r={args.r} modes={args.modes} epochs={args.epochs}\n")

    for ep in range(args.epochs):
        model.train()
        t0 = default_timer()
        train_l2 = 0.0

        perm = torch.randperm(n_train)
        for i in range(0, n_train, args.batch):
            idx = perm[i:i+args.batch]
            xb = x_train[idx]; yb = y_train[idx]; Bb = xb.shape[0]
            g  = grid.unsqueeze(0).expand(Bb, -1, -1, -1)

            pred = model(xb, g, args.T_out)
            loss = paper_rel_l2_loss(pred, yb.permute(0, 3, 1, 2))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_l2 += loss.item() * Bb

        scheduler.step()
        train_l2 /= n_train

        model.eval()
        with torch.no_grad():
            test_l2 = 0.0
            for i in range(0, n_test, args.batch):
                xb = x_test[i:i+args.batch]; yb = y_test[i:i+args.batch]
                Bb = xb.shape[0]
                g  = grid.unsqueeze(0).expand(Bb, -1, -1, -1)
                pred = model(xb, g, args.T_out)
                test_l2 += paper_rel_l2_loss(pred, yb.permute(0, 3, 1, 2)).item() * Bb
            test_l2 /= n_test
            max_re = model.get_max_re_eigenvalue()
            stable = 1 if max_re <= 0 else 0

        t1 = default_timer()

        with open(csv_path, 'a', newline='') as cf:
            csv.writer(cf).writerow([ep+1, f'{train_l2:.6f}', f'{test_l2:.6f}',
                                     f'{max_re:.6f}', stable, f'{t1-t0:.2f}'])

        if (ep + 1) % 10 == 0:
            flag = " <-- BEST" if test_l2 < best_test_l2 else ""
            stab = "stable" if stable else "UNSTABLE!"
            print(f"[{ep+1:04d}] train={train_l2:.4f} test={test_l2:.4f} "
                  f"Re(lam)_max={max_re:.5f} {stab}{flag}")

        if test_l2 < best_test_l2:
            best_test_l2 = test_l2
            torch.save(model.state_dict(), ckpt_path)

    print(f"\n{'='*60}")
    print(f"DONE [DCT-LGN {args.tag}]  best test_l2 = {best_test_l2:.4f}")
    print(f"{'='*60}")
    print(f"Now: python fitzhugh_nagumo/train_fhn_neumann.py --data {args.data} --tag {args.tag} --analyze")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     required=True)
    parser.add_argument('--tag',      default='fhn_dct')
    parser.add_argument('--seed',     type=int,   default=0)
    parser.add_argument('--analyze',  action='store_true')
    parser.add_argument('--T_in',     type=int,   default=10)
    parser.add_argument('--T_out',    type=int,   default=20)
    parser.add_argument('--n_train',  type=int,   default=1000)
    parser.add_argument('--n_test',   type=int,   default=200)
    parser.add_argument('--r',        type=int,   default=32)
    parser.add_argument('--modes',    type=int,   default=12)
    parser.add_argument('--width',    type=int,   default=32)
    parser.add_argument('--epochs',   type=int,   default=500)
    parser.add_argument('--batch',    type=int,   default=10)
    parser.add_argument('--lr',       type=float, default=1e-3)
    parser.add_argument('--save_dir', default='./checkpoints')
    args = parser.parse_args()
    set_seed(args.seed)
    train(args)
