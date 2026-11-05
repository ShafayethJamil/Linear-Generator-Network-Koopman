"""Linearly recurrent autoencoder (LRAN) baseline: the LGN-KM encoder and decoder with an
unconstrained discrete-time latent operator z_{t+1} = A z_t.

Paper: Table 1 (LRAN row: test relative L2, spectral radius of A) and the LRAN rows of Table A.6.

Usage (seeds 0, 1, 2):
    python baselines/lran.py --data data/ns/NavierStokes_V1e-5_N1200_T20.mat --tag lran_nu1e5_s0 --seed 0
    python baselines/lran.py --data data/ns/ns_V1e-3_N5000_T50.mat --T_out 40 --tag lran_nu1e3_s0 --seed 0
    python baselines/lran.py --data <same> --tag <same> --analyze      # eigenvalues of A

Encoder, decoder, data, optimizer and metric are identical to navier_stokes/train_lgnkm.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse, os, csv
from timeit import default_timer

def set_seed(seed):
    """Seed torch (weights, epoch shuffling) and numpy, matching lgn_flexD.py."""
    torch.manual_seed(seed)
    np.random.seed(seed)


# ============================================================
# Spectral Conv 2D / Encoder / Decoder (identical to paper code)
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


# ============================================================
# LRAN model: z_{t+1} = A z_t, A unconstrained (shared across modes)
# ============================================================

class SpatialLRAN(nn.Module):
    def __init__(self, modes, width, r, T_in, H, W):
        super().__init__()
        self.modes, self.r, self.H, self.W = modes, r, H, W
        self.encoder = SpatialEncoder(modes, width, r, T_in)
        self.decoder = SpatialDecoder(r)
        # init near identity: the discrete analogue of expm(small L)
        self.A = nn.Parameter(torch.eye(r) + torch.randn(r, r) * 0.01)

    def propagate(self, z0_ft, t):
        At = torch.linalg.matrix_power(self.A, t)              # (r, r)
        z = z0_ft.permute(0, 2, 3, 1).unsqueeze(-1)            # (B,m,m,r,1)
        z_t = (At.to(torch.cfloat) @ z).squeeze(-1)
        return z_t.permute(0, 3, 1, 2)

    def forward(self, x, grid, T_out, dt=1.0):
        z0    = self.encoder(x, grid)
        z0_ft = torch.fft.rfft2(z0)
        z0_ft_trunc = z0_ft[:, :, :self.modes, :self.modes]

        preds = []
        for t in range(1, T_out + 1):
            zt_ft_trunc = self.propagate(z0_ft_trunc, t)
            zt_ft = z0_ft.clone()
            zt_ft[:, :, :self.modes, :self.modes] = zt_ft_trunc
            z_t = torch.fft.irfft2(zt_ft, s=(self.H, self.W))
            preds.append(self.decoder(z_t))
        return torch.stack(preds, dim=1)

    def eig_stats(self):
        mu = torch.linalg.eigvals(self.A)
        absmu = mu.abs()
        return {
            'max_abs_mu': absmu.max().item(),
            'pct_stable': 100.0 * (absmu <= 1.0 + 1e-9).float().mean().item(),
            'mu': mu.detach().cpu().numpy(),
        }


# ============================================================
# Paper metric / data loading (identical)
# ============================================================

def paper_rel_l2_loss(pred, target):
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = target.reshape(B, -1)
    return ((p - t).norm(dim=1) / (t.norm(dim=1) + 1e-8)).mean()


def eval_per_step(model, x_test, y_test, grid, T_out, batch):
    """Rel-L2 over the full volume AND resolved per rollout step t=1..T_out,
    on the test set. Mirrors the DMD/KNO baselines so all three are directly
    comparable in Table 1 and the long-horizon evaluation."""
    model.eval()
    n_test = x_test.shape[0]
    steps = [[] for _ in range(T_out)]
    tot = []
    with torch.no_grad():
        for i in range(0, n_test, batch):
            xb = x_test[i:i+batch]; yb = y_test[i:i+batch]; Bb = xb.shape[0]
            g = grid.unsqueeze(0).expand(Bb, -1, -1, -1)
            pred = model(xb, g, T_out)              # (B, T_out, H, W)
            targ = yb.permute(0, 3, 1, 2)           # (B, T_out, H, W)
            for t in range(T_out):
                p = pred[:, t].reshape(Bb, -1); q = targ[:, t].reshape(Bb, -1)
                steps[t].append((p - q).norm(dim=1) / (q.norm(dim=1) + 1e-8))
            p = pred.reshape(Bb, -1); q = targ.reshape(Bb, -1)
            tot.append((p - q).norm(dim=1) / (q.norm(dim=1) + 1e-8))
    total = torch.cat(tot).mean().item()
    per_step = [torch.cat(s).mean().item() for s in steps]
    return total, per_step


def write_results(model, tag, save_dir, x_test, y_test, grid, T_out, batch):
    """Final results row: total + per-step rel-L2 + generator stability."""
    total, per_step = eval_per_step(model, x_test, y_test, grid, T_out, batch)
    st = model.eig_stats()
    path = os.path.join(save_dir, f'lran_results_{tag}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['test_rel_l2_full_volume', f'{total:.6f}'])
        for t, v in enumerate(per_step, 1):
            w.writerow([f'l2_step{t}', f'{v:.6f}'])
        w.writerow(['max_abs_mu', f"{st['max_abs_mu']:.6f}"])
        w.writerow(['pct_stable', f"{st['pct_stable']:.2f}"])
        w.writerow(['r', model.r]); w.writerow(['modes', model.modes])
        w.writerow(['T_out', T_out])
    print(f"wrote {path}  total={total:.4f}  "
          f"step1={per_step[0]:.4f} -> step{T_out}={per_step[-1]:.4f}")
    return path


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
    print(f"  Data stats: mean={u.mean():.4f} std={u.std():.4f} (NO normalization)")
    return x_train, y_train, x_test, y_test, grid, H, W


# ============================================================
# Training
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

    model = SpatialLRAN(modes=args.modes, width=args.width, r=args.r,
                        T_in=args.T_in, H=H, W=W).to(device)
    ckpt_path = os.path.join(args.save_dir, f'best_koopman_{args.tag}.pt')

    if args.analyze:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        st = model.eig_stats()
        eig_path = os.path.join(args.save_dir, f'eigenspectrum_{args.tag}.csv')
        with open(eig_path, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['re_mu', 'im_mu', 'abs_mu', 're_log_mu', 'im_log_mu'])
            for m in st['mu']:
                lm = np.log(m) if m != 0 else complex(-np.inf, 0)
                w.writerow([m.real, m.imag, abs(m), lm.real, lm.imag])
        print(f"LRAN [{args.tag}]  max|mu| = {st['max_abs_mu']:.5f}  "
              f"stable = {st['pct_stable']:.1f}%")
        print("NOTE for table: no S-D decomposition, no k-resolved spectrum, "
              "no dispersion readout available from this model by construction.")
        print(f"wrote {eig_path}")
        write_results(model, args.tag, args.save_dir,
                      x_test, y_test, grid, args.T_out, args.batch)
        return

    n_params = sum(p.numel() for p in model.parameters())
    print(f"LRAN params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, f'curve_koopman_{args.tag}.csv')
    with open(csv_path, 'w', newline='') as cf:
        csv.writer(cf).writerow(['epoch', 'train_l2', 'test_l2',
                                 'max_abs_mu', 'stable', 'time_s'])

    best_test_l2 = float('inf')
    print(f"\n[{args.tag}] LRAN (unconstrained A)  T_in={args.T_in} "
          f"T_out={args.T_out} r={args.r} modes={args.modes}\n")

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
            st = model.eig_stats()
            stable = 1 if st['max_abs_mu'] <= 1.0 + 1e-9 else 0

        t1 = default_timer()

        with open(csv_path, 'a', newline='') as cf:
            csv.writer(cf).writerow([ep+1, f'{train_l2:.6f}', f'{test_l2:.6f}',
                                     f"{st['max_abs_mu']:.6f}", stable, f'{t1-t0:.2f}'])

        if (ep + 1) % 10 == 0:
            flag = " <-- BEST" if test_l2 < best_test_l2 else ""
            stab = "stable" if stable else "UNSTABLE"
            print(f"[{ep+1:04d}] train={train_l2:.4f} test={test_l2:.4f} "
                  f"max|mu|={st['max_abs_mu']:.4f} {stab}{flag}")

        if test_l2 < best_test_l2:
            best_test_l2 = test_l2
            torch.save(model.state_dict(), ckpt_path)

    # reload BEST checkpoint (model currently holds last-epoch weights) and
    # write the final per-step results row on it.
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    write_results(model, args.tag, args.save_dir,
                  x_test, y_test, grid, args.T_out, args.batch)

    print(f"\n{'='*60}")
    print(f"DONE [LRAN {args.tag}]  best test_l2 = {best_test_l2:.4f}")
    print(f"{'='*60}")
    print(f"Next: python baselines/lran.py --data {args.data} --tag {args.tag} --save_dir {args.save_dir} --analyze")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     required=True)
    parser.add_argument('--tag',      default='lran_nu1e5')
    parser.add_argument('--seed',     type=int,   default=0,
                        help='random seed (include it in --tag, e.g. _s1)')
    parser.add_argument('--analyze',  action='store_true')
    parser.add_argument('--T_in',     type=int,   default=10)
    parser.add_argument('--T_out',    type=int,   default=10)
    parser.add_argument('--n_train',  type=int,   default=1000)
    parser.add_argument('--n_test',   type=int,   default=200)
    parser.add_argument('--r',        type=int,   default=32)
    parser.add_argument('--modes',    type=int,   default=12)
    parser.add_argument('--width',    type=int,   default=32)
    parser.add_argument('--epochs',   type=int,   default=500)
    parser.add_argument('--batch',    type=int,   default=10)
    parser.add_argument('--lr',       type=float, default=1e-3)
    parser.add_argument('--save_dir', default='outputs/baselines')
    args = parser.parse_args()
    set_seed(args.seed)
    train(args)
