"""
E3 — LONG-HORIZON ACCURACY EVALUATION (suVX W4/Q3, pbkH W2)
============================================================
Rolls LGN-KM, FNO (and optionally LRAN) on the fresh long-horizon truth
(generate_ns_long.py) from the same 10-frame input window, out to t=200.

Metrics written per step t = 11..200:
  - per-sample relative L2 (pointwise accuracy; meaningful up to the
    chaotic predictability horizon, saturates ~O(1) after — for ANY model)
  - total enstrophy  mean(w^2)  of prediction vs truth (statistical accuracy)
  - max |w| (blowup tracking)
At t in {20, 50, 100, 200}: radial energy-spectrum relative error.
Summary: first blowup step per model (max|w| > 1e3 or NaN).

Usage (from the repository root; Table A.6 of the paper):
  python long_horizon/eval_long_horizon.py \
      --data outputs/long_horizon/ns_long_nu1e5_N200_T200_fixed.mat \
      --koopman_ckpt checkpoints/ns/lgnkm_nu1e5_s0.pt --fno_ckpt checkpoints/ns/fno_nu1e5.pt \
      --lran_ckpt checkpoints/baselines/lran_nu1e5_s0.pt --kno_ckpt checkpoints/baselines/kno_nu1e5_s0.pt \
      --save_dir outputs/long_horizon/results/long
Shipped output of that run: long_horizon/results/long/E3_*.csv

Outputs (./checkpoints): E3_per_step.csv, E3_spectrum.csv, E3_summary.csv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import matrix_exp
import numpy as np
import argparse, os, csv

torch.manual_seed(0)
np.random.seed(0)


# ============================================================
# Model classes — verbatim copies so the checkpoints load exactly
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

    def predict_times(self, x, grid, times, dt=1.0):
        """One-shot prediction at each t in `times` (constant cost per query)."""
        z0    = self.encoder(x, grid)
        z0_ft = torch.fft.rfft2(z0)
        z0_ft_trunc = z0_ft[:, :, :self.modes, :self.modes]
        L_all = self.get_generators()

        preds = []
        for t in times:
            eL = matrix_exp(L_all * float(t) * dt)
            z = z0_ft_trunc.permute(0, 2, 3, 1).unsqueeze(-1)
            zt = (eL.unsqueeze(0).to(torch.cfloat) @ z).squeeze(-1).permute(0, 3, 1, 2)
            zt_ft = z0_ft.clone()
            zt_ft[:, :, :self.modes, :self.modes] = zt
            z_t = torch.fft.irfft2(zt_ft, s=(self.H, self.W))
            preds.append(self.decoder(z_t))
        return torch.stack(preds, dim=1)                    # (B, len(times), H, W)


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
            p = torch.nan_to_num(p, nan=1e6, posinf=1e6, neginf=-1e6)  # keep rolling past blowup
            preds.append(p)
            x = torch.cat([x[:, :, :, 1:], p.unsqueeze(-1)], dim=-1)
        return torch.stack(preds, dim=1)


class SpatialLRAN(nn.Module):
    def __init__(self, modes, width, r, T_in, H, W):
        super().__init__()
        self.modes, self.r, self.H, self.W = modes, r, H, W
        self.encoder = SpatialEncoder(modes, width, r, T_in)
        self.decoder = SpatialDecoder(r)
        self.A = nn.Parameter(torch.eye(r) + torch.randn(r, r) * 0.01)

    def predict_times(self, x, grid, times):
        z0    = self.encoder(x, grid)
        z0_ft = torch.fft.rfft2(z0)
        z0_ft_trunc = z0_ft[:, :, :self.modes, :self.modes]
        preds = []
        for t in times:
            At = torch.linalg.matrix_power(self.A, int(t))
            z = z0_ft_trunc.permute(0, 2, 3, 1).unsqueeze(-1)
            zt = (At.to(torch.cfloat) @ z).squeeze(-1).permute(0, 3, 1, 2)
            zt_ft = z0_ft.clone()
            zt_ft[:, :, :self.modes, :self.modes] = zt
            z_t = torch.fft.irfft2(zt_ft, s=(self.H, self.W))
            preds.append(self.decoder(z_t))
        return torch.stack(preds, dim=1)


class KNORollout:
    """Wraps a trained koopmanlab KNO2d kernel with an FNO-style autoregressive
    rollout. KNO is a discrete per-step operator (not a continuous-time
    generator) and is unconstrained (no stability guarantee), so like FNO it is
    rolled frame-by-frame and can diverge at long horizon — nan_to_num keeps it
    rolling past blowup so the blowup step is recorded rather than crashing.
    forward returns (prediction, reconstruction); the last channel is next-step."""
    def __init__(self, kernel):
        self.kernel = kernel

    def eval(self):
        self.kernel.eval(); return self

    def rollout(self, x0, grid, steps):        # grid ignored: KNO2d takes no grid
        preds = []
        cur = x0.clone()                       # (B,H,W,T_in)
        for _ in range(steps):
            im, _ = self.kernel(cur)           # (B,H,W,T_in), (B,H,W,T_in)
            out = torch.nan_to_num(im[..., -1:], nan=1e6, posinf=1e6, neginf=-1e6)
            preds.append(out.squeeze(-1))      # (B,H,W)
            cur = torch.cat([cur[..., 1:], out], dim=-1)   # slide window (width T_in)
        return torch.stack(preds, dim=1)       # (B, steps, H, W)


# ============================================================
# Metrics
# ============================================================

def radial_spectrum(w):
    """w: (B, H, W) -> (B, n_bins) radial-binned |fft|^2."""
    B, H, W = w.shape
    wh = np.fft.fft2(w, axes=(1, 2))
    p = np.abs(wh) ** 2
    k = np.fft.fftfreq(H, d=1.0/H)
    k1, k2 = np.meshgrid(k, k, indexing='ij')
    kr = np.sqrt(k1**2 + k2**2)
    bins = np.arange(0, H//2 + 1)
    idx = np.digitize(kr.ravel(), bins)
    spec = np.zeros((B, len(bins)))
    for b in range(B):
        pb = p[b].ravel()
        for i in range(1, len(bins) + 1):
            m = idx == i
            if m.any():
                spec[b, i-1] = pb[m].sum()
    return spec


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    import h5py
    with h5py.File(args.data, 'r') as f:
        u = np.array(f['u'], dtype=np.float32)
        t_arr = np.array(f['t']) if 't' in f else None

    if t_arr is not None:
        # generate_ns_long.py output: layout is (N, H, W, T) by construction.
        # Verify against the saved time vector instead of guessing — the usual
        # "shape[0] < shape[-1]" heuristic MISFIRES here (N=50 < T=200 would
        # transpose time onto the batch axis and silently corrupt the eval).
        assert u.ndim == 4 and u.shape[-1] == len(t_arr), \
            f"time-axis mismatch: u{u.shape} vs {len(t_arr)} recorded times"
    else:
        # Legacy files without a time vector: (N,H,W,T) vs (T,H,W,N) is
        # fundamentally ambiguous from shape alone (e.g. 50 vs 200 could be
        # either). Only transpose in the unambiguous MATLAB-export case:
        # first axis is a small frame count AND last axis is a large batch.
        if u.ndim == 4 and u.shape[0] < 64 and u.shape[-1] >= 1000:
            print(f"  NOTE: transposing legacy layout {u.shape} -> (N,H,W,T)")
            u = np.transpose(u, (3, 1, 2, 0))
    N, H, W, T = u.shape
    print(f"Truth: N={N} H={H} W={W} T={T} (frames at t=1..{T})")

    T_in = 10
    n_pred = T - T_in                                       # steps 11..T
    x_in = torch.from_numpy(u[:, :, :, :T_in]).to(device)   # (N,H,W,10)
    truth = u[:, :, :, T_in:]                               # (N,H,W,n_pred)

    gx = torch.linspace(0, 1, H); gy = torch.linspace(0, 1, W)
    grid = torch.stack(torch.meshgrid(gx, gy, indexing='ij'), dim=-1).to(device)

    models = {}
    lgn = SpatialKoopmanLGN(modes=12, width=32, r=32, T_in=T_in, H=H, W=W).to(device)
    lgn.load_state_dict(torch.load(args.koopman_ckpt, map_location=device))
    lgn.eval(); models['lgn'] = lgn

    fno = FNO2d(modes1=12, modes2=12, width=20, T_in=T_in).to(device)
    fno.load_state_dict(torch.load(args.fno_ckpt, map_location=device))
    fno.eval(); models['fno'] = fno

    if args.lran_ckpt:
        lran = SpatialLRAN(modes=12, width=32, r=32, T_in=T_in, H=H, W=W).to(device)
        lran.load_state_dict(torch.load(args.lran_ckpt, map_location=device))
        lran.eval(); models['lran'] = lran

    if args.kno_ckpt:
        import koopmanlab as kp
        ckpt = torch.load(args.kno_ckpt, map_location=device)
        cfg = ckpt['config']                              # saved by baselines/kno.py
        kno = kp.model.koopman(backbone=cfg['backbone'], autoencoder=cfg['autoencoder'],
                               o=cfg['o'], m=cfg['m'], r=cfg['r_power'],
                               t_in=cfg['T_in'], device=device)
        kno.compile()
        kno.kernel.load_state_dict(ckpt['state_dict'])
        models['kno'] = KNORollout(kno.kernel).eval()

    times = list(range(1, n_pred + 1))
    preds = {}
    with torch.no_grad():
        for name, m in models.items():
            chunks = []
            for i in range(0, N, args.batch):
                xb = x_in[i:i+args.batch]
                g = grid.unsqueeze(0).expand(xb.shape[0], -1, -1, -1)
                if name in ('fno', 'kno'):
                    p = m.rollout(xb, g, n_pred)
                else:
                    p = m.predict_times(xb, g, times)
                chunks.append(p.cpu().numpy())
            preds[name] = np.concatenate(chunks, axis=0)    # (N, n_pred, H, W)
            print(f"{name}: rolled {n_pred} steps")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- per-step metrics ----
    per_step_path = os.path.join(args.save_dir, 'E3_per_step.csv')
    names = list(preds.keys())
    with open(per_step_path, 'w', newline='') as f:
        w = csv.writer(f)
        hdr = ['t', 'true_enstrophy']
        for n in names:
            hdr += [f'{n}_rel_l2', f'{n}_enstrophy', f'{n}_max_abs']
        w.writerow(hdr)
        blowup = {n: None for n in names}
        for s in range(n_pred):
            t_phys = T_in + s + 1
            tr = truth[:, :, :, s]                          # (N,H,W)
            row = [t_phys, f'{np.mean(tr**2):.6f}']
            for n in names:
                pr = preds[n][:, s]
                num = np.linalg.norm((pr - tr).reshape(N, -1), axis=1)
                den = np.linalg.norm(tr.reshape(N, -1), axis=1) + 1e-8
                rel = float(np.mean(num / den))
                ens = float(np.mean(pr ** 2))
                mx  = float(np.abs(pr).max())
                row += [f'{rel:.6f}', f'{ens:.6f}', f'{mx:.4e}']
                if blowup[n] is None and (not np.isfinite(mx) or mx > 1e3):
                    blowup[n] = t_phys
            w.writerow(row)
    print(f"wrote {per_step_path}")

    # ---- spectrum errors at checkpoints ----
    spec_path = os.path.join(args.save_dir, 'E3_spectrum.csv')
    checks = [t for t in [20, 50, 100, 200] if t <= T]
    with open(spec_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t'] + [f'{n}_spectrum_rel_err' for n in names])
        for t_phys in checks:
            s = t_phys - T_in - 1
            if s < 0:
                continue
            S_true = radial_spectrum(truth[:, :, :, s]).mean(axis=0)
            row = [t_phys]
            for n in names:
                pr = np.nan_to_num(preds[n][:, s], nan=0.0, posinf=0.0, neginf=0.0)
                S_p = radial_spectrum(pr).mean(axis=0)
                err = float(np.abs(S_p - S_true).sum() / (S_true.sum() + 1e-12))
                row.append(f'{err:.6f}')
            w.writerow(row)
    print(f"wrote {spec_path}")

    # ---- summary ----
    summ_path = os.path.join(args.save_dir, 'E3_summary.csv')
    with open(summ_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'first_blowup_t (max|w|>1e3)', 'rel_l2_t20', 'rel_l2_final'])
        with open(per_step_path) as pf:
            rows = list(csv.DictReader(pf))
        for n in names:
            r20 = next((r[f'{n}_rel_l2'] for r in rows if int(r['t']) == 20), '')
            rf  = rows[-1][f'{n}_rel_l2']
            w.writerow([n, blowup[n] if blowup[n] else 'never', r20, rf])
    print(f"wrote {summ_path}")

    print("\nSummary:")
    for n in names:
        print(f"  {n}: blowup={'never' if not blowup[n] else 't='+str(blowup[n])}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data',         required=True)
    p.add_argument('--koopman_ckpt', required=True)
    p.add_argument('--fno_ckpt',     required=True)
    p.add_argument('--lran_ckpt',    default=None)
    p.add_argument('--kno_ckpt',     default=None,
                   help='trained koopmanlab KNO checkpoint (best_kno_*.pt from baselines/kno.py)')
    p.add_argument('--batch',        type=int, default=5)
    p.add_argument('--save_dir',     default='./checkpoints')
    args = p.parse_args()
    main(args)
