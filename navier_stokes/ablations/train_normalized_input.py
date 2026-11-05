"""
Release: originally LGN_koopman/koopman_spatial.py. It trained the "normalized input" row of Table A.8
(Appendix A.10): z-scored inputs, a per-frame relative-L2 loss, 1000 epochs. This is NOT the paper's main
protocol (navier_stokes/train_lgnkm.py: raw data, full-volume loss, 500 epochs). Command reconstructed
from the released curve (1000 epochs) and the nu = 1e-5 data length (T = 20, so T_out = 10):
    python navier_stokes/ablations/train_normalized_input.py --data data/ns/NavierStokes_V1e-5_N1200_T20.mat \
        --T_in 10 --T_out 10 --epochs 1000 --tag nu1e5_spatial --save_dir runs/normalized
Released checkpoint: checkpoints/ns_ablations/sensitivity/normalized_input.pt. The physical interpretation in
the original note below (alpha proportional to nu, zero-shot transfer) is a design hypothesis, not a paper claim.

Original docstring:
Spatial Koopman-LGN for 2D Navier-Stokes
==========================================
Key fix over v1: latent state is spatial z(x,y) in R^(r,H,W)
NOT a global vector. Full spatial structure preserved.

Generator architecture (physically motivated):
    For each Fourier mode k=(k1,k2):
        L_k = S - D_k
        S    = (P - P^T)/2          skew-symmetric, SHARED across modes
                                     encodes inter-channel energy transfer
        D_k  = diag(softplus(d) + softplus(alpha) * |k|^2 / kmax^2)
                                     mode-dependent dissipation
                                     alpha*|k|^2 ~ viscous decay rate nu*|k|^2

Physical interpretation:
    - S: nonlinear advection couples vorticity modes (energy cascade)
    - D_k: viscous dissipation — higher modes decay faster
    - alpha recoverable = proportional to nu (zero-shot viscosity transfer)

Stability: Re(eigenvalues(L_k)) <= 0 for ALL k  GUARANTEED
Inverse design: gradient flows through expm(L_k * t) exactly

Usage:
    CUDA_VISIBLE_DEVICES=0 python koopman_spatial.py \\
        --data ns_V1e-3_N5000_T50.mat \\
        --T_in 10 --T_out 40 --epochs 500 --tag nu1e3_spatial
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import matrix_exp
import numpy as np
import h5py
import argparse, os, csv
from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)


# ============================================================
# Spectral Conv 2D (standard FNO block — encoder/decoder)
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


# ============================================================
# Encoder: omega(B,H,W,T_in) -> z0(B,r,H,W)   SPATIAL latent
# ============================================================

class SpatialEncoder(nn.Module):
    def __init__(self, modes, width, r, T_in):
        super().__init__()
        # Lift: T_in+2 -> width channels
        self.fc0    = nn.Linear(T_in + 2, width)
        # 2 spectral conv layers
        self.conv0  = SpectralConv2d(width, width, modes, modes)
        self.conv1  = SpectralConv2d(width, width, modes, modes)
        self.w0     = nn.Conv2d(width, width, 1)
        self.w1     = nn.Conv2d(width, width, 1)
        # Project to latent dim r — keep spatial (B, r, H, W)
        self.proj   = nn.Conv2d(width, r, 1)

    def forward(self, x, grid):
        # x: (B,H,W,T_in)  grid: (B,H,W,2)
        x = torch.cat([x, grid], dim=-1)        # (B,H,W,T_in+2)
        x = self.fc0(x).permute(0, 3, 1, 2)    # (B,width,H,W)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        return self.proj(x)                      # (B, r, H, W)


# ============================================================
# Decoder: z(B,r,H,W) -> omega(B,H,W)   pointwise
# ============================================================

class SpatialDecoder(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(r, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, z):
        return self.net(z).squeeze(1)       # (B, H, W)


# ============================================================
# Spatial Koopman-LGN
# ============================================================

class SpatialKoopmanLGN(nn.Module):
    """
    Latent evolution in Fourier space:

        F(z0)[k] in C^r  for each mode k=(k1,k2)
        F(z_t)[k] = expm(L_k * t) @ F(z0)[k]

    Generator (shared S, mode-dependent D):
        L_k = S - D_k
        S   = (P - P^T)/2                     skew-symmetric
        D_k = diag(softplus(d) + softplus(a)*|k|^2/kmax^2)

    Stack all L_k into (n_modes, r, r) and use batched matrix_exp.
    """
    def __init__(self, modes, width, r, T_in, H, W):
        super().__init__()
        self.modes = modes
        self.r     = r
        self.H     = H
        self.W     = W

        self.encoder = SpatialEncoder(modes, width, r, T_in)
        self.decoder = SpatialDecoder(r)

        # Generator parameters
        self.P     = nn.Parameter(torch.randn(r, r) * 0.01)   # -> skew S
        self.d     = nn.Parameter(torch.full((r,), -2.0))      # base dissipation
        self.alpha = nn.Parameter(torch.full((r,), -3.0))      # wavenumber scaling

        # Precompute |k|^2 for all modes — register as buffer
        k1 = torch.arange(modes, dtype=torch.float32)
        k2 = torch.arange(modes, dtype=torch.float32)
        kk1, kk2 = torch.meshgrid(k1, k2, indexing='ij')
        # Normalize by kmax^2 so values in [0,1]
        k2_norm = (kk1**2 + kk2**2) / (modes**2)    # (modes, modes)
        self.register_buffer('k2_norm', k2_norm)     # (modes, modes)

    def get_generators(self):
        """
        Returns L_all: (modes, modes, r, r)  — one generator per Fourier mode
        All have Re(eigenvalues) <= 0  GUARANTEED
        """
        S   = (self.P - self.P.T) / 2.0                      # (r, r) skew
        d0  = F.softplus(self.d)                              # (r,) base dissipation
        a0  = F.softplus(self.alpha)                          # (r,) wavenumber scaling

        # D_k = diag(d0 + a0 * |k|^2)  shape: (modes, modes, r)
        # k2_norm: (modes, modes) -> unsqueeze -> (modes, modes, 1)
        D_diag = d0.unsqueeze(0).unsqueeze(0) + \
                 a0.unsqueeze(0).unsqueeze(0) * self.k2_norm.unsqueeze(-1)
        # (modes, modes, r)

        # Expand S to (modes, modes, r, r) and subtract D on diagonal
        S_exp  = S.unsqueeze(0).unsqueeze(0).expand(
                     self.modes, self.modes, -1, -1).clone()  # (modes, modes, r, r)
        D_mat  = torch.diag_embed(D_diag)                    # (modes, modes, r, r)
        L_all  = S_exp - D_mat                               # (modes, modes, r, r)
        return L_all

    def propagate(self, z0_ft, t, dt=1.0):
        """
        z0_ft: (B, r, modes, modes) complex  — Fourier coefficients
        Returns z_t_ft: same shape
        """
        L_all = self.get_generators()                        # (modes, modes, r, r)
        # Compute batched matrix exponential: (modes, modes, r, r)
        eL = matrix_exp(L_all * t * dt)                     # (modes, modes, r, r)

        # Apply to Fourier coefficients
        # z0_ft: (B, r, modes, modes) -> rearrange to (B, modes, modes, r)
        z = z0_ft.permute(0, 2, 3, 1)                       # (B, modes, modes, r)

        # Matrix multiply: z_t[b,k1,k2] = eL[k1,k2] @ z[b,k1,k2]
        # eL: (modes, modes, r, r)  z: (B, modes, modes, r, 1)
        z = z.unsqueeze(-1)                                  # (B, modes, modes, r, 1)
        eL_exp = eL.unsqueeze(0)                             # (1, modes, modes, r, r)
        z_t = (eL_exp.to(torch.cfloat) @ z).squeeze(-1)                      # (B, modes, modes, r)
        return z_t.permute(0, 3, 1, 2)                      # (B, r, modes, modes)

    def forward(self, x, grid, T_out, dt=1.0):
        """
        x:     (B, H, W, T_in)
        grid:  (B, H, W, 2)
        Returns: (B, T_out, H, W)
        """
        z0    = self.encoder(x, grid)                    # (B, r, H, W)
        z0_ft = torch.fft.rfft2(z0)                      # (B, r, H, W//2+1) complex
        # Take only the first [modes x modes] block we parameterize
        z0_ft_trunc = z0_ft[:, :, :self.modes, :self.modes]  # (B, r, modes, modes)

        preds = []
        for t in range(1, T_out + 1):
            # Evolve truncated modes
            zt_ft_trunc = self.propagate(z0_ft_trunc, t, dt)  # (B, r, modes, modes)

            # Put back into full FFT array
            zt_ft = z0_ft.clone()
            zt_ft[:, :, :self.modes, :self.modes] = zt_ft_trunc

            # iFFT -> spatial latent
            z_t = torch.fft.irfft2(zt_ft, s=(self.H, self.W))  # (B, r, H, W)

            # Decode
            omega_t = self.decoder(z_t)                          # (B, H, W)
            preds.append(omega_t)

        return torch.stack(preds, dim=1)                         # (B, T_out, H, W)

    def get_max_re_eigenvalue(self):
        """Check stability across all modes — max Re(eigenvalue) should be <= 0"""
        L_all = self.get_generators()
        # Sample a few modes for speed
        L_flat = L_all.reshape(-1, self.r, self.r)
        eigs = torch.linalg.eigvals(L_flat)
        return eigs.real.max().item()


# ============================================================
# Loss
# ============================================================

def rel_l2(pred, target):
    return ((pred - target).norm(dim=(-2,-1)) /
            (target.norm(dim=(-2,-1)) + 1e-8)).mean()


# ============================================================
# Data loading
# ============================================================

def load_data(mat_path, T_in=10, T_out=40, n_train=1000, n_test=200, target_res=64):
    print(f"Loading {mat_path} ...")
    try:
        with h5py.File(mat_path, 'r') as f:
            keys = list(f.keys())
            print(f"  Format: HDF5  Keys: {keys}")
            key = 'u' if 'u' in keys else keys[0]
            arr = np.array(f[key], dtype=np.float32)
    except OSError:
        import scipy.io
        data = scipy.io.loadmat(mat_path)
        keys = [k for k in data.keys() if not k.startswith('_')]
        print(f"  Format: v5  Keys: {keys}")
        key = 'u' if 'u' in keys else keys[0]
        arr = data[key].astype(np.float32)
    print(f"  Raw shape: {arr.shape}")

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
        u = F.interpolate(u, size=(target_res, target_res), mode='bilinear', align_corners=False)
        u = u.reshape(N, T, target_res, target_res).permute(0, 2, 3, 1)
        H, W = target_res, target_res

    if T < T_in + T_out:
        raise ValueError(f"T={T} < T_in+T_out={T_in+T_out}")

    n_train = min(n_train, N - n_test)
    n_test  = min(n_test,  N - n_train)

    gx   = torch.linspace(0, 1, H)
    gy   = torch.linspace(0, 1, W)
    grid = torch.stack(torch.meshgrid(gx, gy, indexing='ij'), dim=-1)

    x_train = u[:n_train,               :, :, :T_in]
    y_train = u[:n_train,               :, :, T_in:T_in+T_out]
    x_test  = u[n_train:n_train+n_test, :, :, :T_in]
    y_test  = u[n_train:n_train+n_test, :, :, T_in:T_in+T_out]

    mean = x_train.mean(); std = x_train.std() + 1e-8
    x_train=(x_train-mean)/std; y_train=(y_train-mean)/std
    x_test=(x_test-mean)/std;   y_test=(y_test-mean)/std

    print(f"  Train: {x_train.shape} -> {y_train.shape}  n_train={n_train} n_test={n_test}")
    return x_train, y_train, x_test, y_test, grid, H, W


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    x_train, y_train, x_test, y_test, grid, H, W = load_data(
        args.data, args.T_in, args.T_out, args.n_train, args.n_test)

    x_train,y_train = x_train.to(device), y_train.to(device)
    x_test, y_test  = x_test.to(device),  y_test.to(device)
    grid = grid.to(device)
    n_train, n_test = x_train.shape[0], x_test.shape[0]

    model = SpatialKoopmanLGN(
        modes=args.modes, width=args.width, r=args.r,
        T_in=args.T_in, H=H, W=W).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Spatial Koopman-LGN params: {n_params:,}")
    print(f"  Generator: {args.modes}x{args.modes} modes, r={args.r} -> "
          f"{args.modes**2} independent L_k matrices of size {args.r}x{args.r}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, f'curve_{args.tag}.csv')
    with open(csv_path, 'w', newline='') as cf:
        csv.writer(cf).writerow(['epoch','train_l2','test_l2','max_re_eig','stable','time_s'])

    best_test_l2 = float('inf')
    print(f"\n[{args.tag}] T_in={args.T_in} T_out={args.T_out} "
          f"r={args.r} modes={args.modes} epochs={args.epochs}\n")

    for ep in range(args.epochs):
        model.train()
        t0 = default_timer()
        train_l2 = 0.0

        perm = torch.randperm(n_train)
        for i in range(0, n_train, args.batch):
            idx = perm[i:i+args.batch]
            xb  = x_train[idx]; yb = y_train[idx]; Bb = xb.shape[0]
            g   = grid.unsqueeze(0).expand(Bb, -1, -1, -1)

            pred = model(xb, g, args.T_out)
            loss = rel_l2(pred, yb.permute(0, 3, 1, 2))

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
                test_l2 += rel_l2(pred, yb.permute(0, 3, 1, 2)).item() * Bb
            test_l2 /= n_test
            max_re  = model.get_max_re_eigenvalue()
            stable  = 1 if max_re <= 0 else 0

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
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f'best_{args.tag}.pt'))

    print(f"\n{'='*55}")
    print(f"DONE [{args.tag}]  best test_l2 = {best_test_l2:.4f}")
    print(f"Comparison:")
    print(f"  FNO-2D (paper) nu=1e-3:  0.0128")
    print(f"  FNO-2D (ours)  nu=1e-3:  0.0140")
    print(f"  Koopman-LGN v1 nu=1e-5:  0.2509  (global pooling)")
    print(f"  Koopman-LGN v2 [{args.tag}]: {best_test_l2:.4f}  (spatial)")
    print(f"{'='*55}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     required=True)
    parser.add_argument('--tag',      default='spatial')
    parser.add_argument('--T_in',     type=int,   default=10)
    parser.add_argument('--T_out',    type=int,   default=40)
    parser.add_argument('--n_train',  type=int,   default=1000)
    parser.add_argument('--n_test',   type=int,   default=200)
    parser.add_argument('--r',        type=int,   default=32,
                        help='Latent channels (smaller than v1 — spatial so memory scales HxW)')
    parser.add_argument('--modes',    type=int,   default=12)
    parser.add_argument('--width',    type=int,   default=32)
    parser.add_argument('--epochs',   type=int,   default=500)
    parser.add_argument('--batch',    type=int,   default=10,
                        help='Smaller batch — spatial latent uses more memory')
    parser.add_argument('--lr',       type=float, default=1e-3)
    parser.add_argument('--save_dir', default='./checkpoints')
    args = parser.parse_args()
    train(args)
