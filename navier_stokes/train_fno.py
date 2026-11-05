"""Train the FNO-2D baseline on 2D Navier-Stokes.

Paper: Table 1, Fig. 5, Appendix A.1 (checkpoints/ns/fno_nu1e{5,3}.pt).

Usage:
    python navier_stokes/train_fno.py --data data/ns/NavierStokes_V1e-5_N1200_T20.mat --T_out 10 --tag nu1e5
    python navier_stokes/train_fno.py --data data/ns/ns_V1e-3_N5000_T50.mat --T_out 40 --tag nu1e3

Autoregressive rollout in training and test; test L2 is evaluated every 10 epochs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
import argparse, os, csv
from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)

# ============================================================
# Spectral Convolution
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
# FNO-2D
# ============================================================

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
# Paper metric — per-sample rel L2 over full spatiotemporal volume
# ============================================================

def paper_rel_l2_loss(pred, target):
    """
    pred, target: (B, T_out, H, W)
    Returns scalar: mean over samples of ||pred-target||/||target||
    """
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = target.reshape(B, -1)
    return ((p - t).norm(dim=1) / (t.norm(dim=1) + 1e-8)).mean()


# ============================================================
# Data loading — NO normalization
# ============================================================

def load_data(mat_path, T_in=10, T_out=10, n_train=1000, n_test=200):
    print(f"Loading {mat_path} ...")
    try:
        data = scipy.io.loadmat(mat_path)
        print("  Format: scipy v5")
        u = torch.from_numpy(data['u'].astype(np.float32))
    except Exception:
        import h5py
        print("  Format: HDF5 v7.3")
        with h5py.File(mat_path, 'r') as f:
            arr = np.array(f['u'], dtype=np.float32)
            u = torch.from_numpy(arr)
        if u.dim() == 4 and u.shape[0] < u.shape[-1]:
            u = u.permute(3, 1, 2, 0)

    N, H, W, T = u.shape
    print(f"  Shape: N={N} H={H} W={W} T={T}")
    assert T >= T_in + T_out, f"T={T} < T_in+T_out={T_in+T_out}"

    gx = torch.linspace(0, 1, H)
    gy = torch.linspace(0, 1, W)
    grid = torch.stack(torch.meshgrid(gx, gy, indexing='ij'), dim=-1)

    # NO NORMALIZATION — raw data
    x_train = u[:n_train,               :, :, :T_in]
    y_train = u[:n_train,               :, :, T_in:T_in+T_out]
    x_test  = u[n_train:n_train+n_test, :, :, :T_in]
    y_test  = u[n_train:n_train+n_test, :, :, T_in:T_in+T_out]

    print(f"  Train: {x_train.shape} -> {y_train.shape}")
    print(f"  Test:  {x_test.shape}  -> {y_test.shape}")
    print(f"  Data stats: mean={u.mean():.4f} std={u.std():.4f} (NO normalization applied)")
    return x_train, y_train, x_test, y_test, grid


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    x_train, y_train, x_test, y_test, grid = load_data(
        args.data, args.T_in, args.T_out, args.n_train, args.n_test)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test,  y_test  = x_test.to(device),  y_test.to(device)
    grid = grid.to(device)
    n_train, n_test = x_train.shape[0], x_test.shape[0]

    model = FNO2d(modes1=args.modes, modes2=args.modes,
                  width=args.width, T_in=args.T_in).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FNO params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, f'curve_fno_{args.tag}.csv')
    with open(csv_path, 'w', newline='') as cf:
        csv.writer(cf).writerow(['epoch', 'train_l2', 'test_l2', 'time_s'])

    best_test_l2 = float('inf')
    print(f"\nTraining FNO: modes={args.modes}, width={args.width}, "
          f"epochs={args.epochs}, batch={args.batch}\n")

    for ep in range(args.epochs):
        model.train()
        t0 = default_timer()
        train_l2 = 0.0

        perm = torch.randperm(n_train)
        x_train, y_train = x_train[perm], y_train[perm]

        for i in range(0, n_train, args.batch):
            xb = x_train[i:i+args.batch]
            yb = y_train[i:i+args.batch]
            Bb = xb.shape[0]
            g  = grid.unsqueeze(0).expand(Bb, -1, -1, -1)

            m = model.module if hasattr(model, 'module') else model
            pred = m.rollout(xb, g, args.T_out)        # (B, T_out, H, W)
            yb_t = yb.permute(0, 3, 1, 2)              # (B, T_out, H, W)

            loss = paper_rel_l2_loss(pred, yb_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_l2 += loss.item() * Bb

        scheduler.step()
        train_l2 /= n_train

        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_l2 = 0.0
                for i in range(0, n_test, args.batch):
                    xb = x_test[i:i+args.batch]
                    yb = y_test[i:i+args.batch]
                    Bb = xb.shape[0]
                    g  = grid.unsqueeze(0).expand(Bb, -1, -1, -1)
                    m  = model.module if hasattr(model, 'module') else model
                    pred = m.rollout(xb, g, args.T_out)
                    yb_t = yb.permute(0, 3, 1, 2)
                    test_l2 += paper_rel_l2_loss(pred, yb_t).item() * Bb
                test_l2 /= n_test

            t1 = default_timer()
            flag = " *** BEST" if test_l2 < best_test_l2 else ""
            print(f"[{ep+1:04d}/{args.epochs}] "
                  f"train_l2={train_l2:.4f}  test_l2={test_l2:.4f}  "
                  f"time={t1-t0:.1f}s{flag}")

            with open(csv_path, 'a', newline='') as cf:
                csv.writer(cf).writerow([ep+1, f'{train_l2:.6f}', f'{test_l2:.6f}', f'{t1-t0:.1f}'])

            if test_l2 < best_test_l2:
                best_test_l2 = test_l2
                m = model.module if hasattr(model, 'module') else model
                torch.save(m.state_dict(), f'{args.save_dir}/best_fno_{args.tag}.pt')

    print(f"\n{'='*60}")
    print(f"DONE [FNO {args.tag}]  best test_l2 = {best_test_l2:.4f}")
    print(f"Paper target (FNO-3D nu=1e-5): 0.1893")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     required=True)
    parser.add_argument('--tag',      default='nu1e5_raw')
    parser.add_argument('--T_in',     type=int,   default=10)
    parser.add_argument('--T_out',    type=int,   default=10)
    parser.add_argument('--n_train',  type=int,   default=1000)
    parser.add_argument('--n_test',   type=int,   default=200)
    parser.add_argument('--modes',    type=int,   default=12)
    parser.add_argument('--width',    type=int,   default=20)
    parser.add_argument('--epochs',   type=int,   default=500)
    parser.add_argument('--batch',    type=int,   default=20)
    parser.add_argument('--save_dir', default='outputs/fno')
    args = parser.parse_args()
    train(args)
