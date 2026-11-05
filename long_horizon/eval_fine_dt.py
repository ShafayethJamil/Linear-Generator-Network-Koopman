"""
E4 — CONTINUOUS-TIME EVAL AT UNSEEN TEMPORAL RESOLUTIONS (suVX W5)
===================================================================
Loads the paper's LGN-KM checkpoint (trained ONLY on the integer dt=1 grid)
and evaluates it at fractional times t = 11.25 ... 19.75 against fine-dt
ground truth (generate_ns_finedt.py), via a single matrix exponential
exp(L·t) per query — no retraining, no interpolation.

Baselines at the same fractional times:
  - interp_lgn  : linear interpolation between the model's OWN predictions
                  at the adjacent integer times (what a practitioner would
                  do with any fixed-step model, incl. FNO).
  - interp_true : linear interpolation between the adjacent integer-time
                  GROUND-TRUTH frames (oracle; measures how much genuine
                  sub-frame structure exists to be captured at all).

If direct exp(L·t) beats interp_lgn and approaches interp_true's gap, the
continuous-time parameterization carries real temporal information beyond
the training grid.

Usage (from the repository root; Table A.7 of the paper):
  python long_horizon/eval_fine_dt.py \
      --data outputs/long_horizon/ns_finedt_nu1e5_N200_T20_fixed.mat \
      --koopman_ckpt checkpoints/ns/lgnkm_nu1e5_s0.pt --save_dir outputs/long_horizon/results/finedt
Shipped output of that run: long_horizon/results/finedt/E4_*.csv

Outputs (./checkpoints): E4_fine_dt.csv (per query time), E4_summary.csv
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
# Model classes — verbatim copies so the checkpoint loads exactly
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
        """One-shot prediction at each (possibly fractional) t in `times`."""
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


# ============================================================
# Eval
# ============================================================

def rel_l2(pred, true):
    """pred, true: (N, H, W) -> mean per-sample relative L2."""
    N = pred.shape[0]
    num = np.linalg.norm((pred - true).reshape(N, -1), axis=1)
    den = np.linalg.norm(true.reshape(N, -1), axis=1) + 1e-8
    return float(np.mean(num / den))


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    import h5py
    with h5py.File(args.data, 'r') as f:
        u = np.array(f['u'], dtype=np.float32)              # (N, H, W, n_frames)
        t_arr = np.array(f['t'], dtype=np.float64)
    assert u.ndim == 4 and u.shape[-1] == len(t_arr), \
        f"time-axis mismatch: u{u.shape} vs {len(t_arr)} recorded times"
    N, H, W, _ = u.shape
    dt_rec = float(t_arr[1] - t_arr[0])
    print(f"Truth: N={N} {H}x{W}, frames at t={t_arr[0]:.2f}..{t_arr[-1]:.2f} step {dt_rec}")

    def frame_at(tau):
        idx = int(round(tau / dt_rec)) - 1
        assert abs(t_arr[idx] - tau) < 1e-6, f"no frame at t={tau}"
        return u[:, :, :, idx]

    T_in = 10
    # input window: integer frames t = 1..10 (exactly the training convention)
    x_in = np.stack([frame_at(t) for t in range(1, T_in + 1)], axis=-1)  # (N,H,W,10)
    x_in = torch.from_numpy(x_in).to(device)

    gx = torch.linspace(0, 1, H); gy = torch.linspace(0, 1, W)
    grid = torch.stack(torch.meshgrid(gx, gy, indexing='ij'), dim=-1).to(device)

    model = SpatialKoopmanLGN(modes=12, width=32, r=32, T_in=T_in, H=H, W=W).to(device)
    model.load_state_dict(torch.load(args.koopman_ckpt, map_location=device))
    model.eval()

    # query times: fractional taus strictly inside (11, 20); integers for sanity
    n_sub = int(round(1.0 / dt_rec))                        # 4 for dt_rec=0.25
    frac_taus = [11 + i * dt_rec for i in range(1, (int(t_arr[-1]) - 11) * n_sub)
                 if abs((11 + i * dt_rec) - round(11 + i * dt_rec)) > 1e-6]
    int_taus  = list(range(11, int(t_arr[-1]) + 1))

    with torch.no_grad():
        g = grid.unsqueeze(0).expand(N, -1, -1, -1)
        # direct continuous-time queries (model time = tau - T_in)
        pred_frac = model.predict_times(x_in, g, [tau - T_in for tau in frac_taus]
                                        ).cpu().numpy()     # (N, n_frac, H, W)
        pred_int  = model.predict_times(x_in, g, [tau - T_in for tau in int_taus]
                                        ).cpu().numpy()     # (N, n_int, H, W)

    int_index = {tau: i for i, tau in enumerate(int_taus)}

    os.makedirs(args.save_dir, exist_ok=True)
    per_path = os.path.join(args.save_dir, 'E4_fine_dt.csv')
    rows = []
    with open(per_path, 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['tau', 'phase', 'lgn_direct', 'interp_lgn', 'interp_true'])
        for j, tau in enumerate(frac_taus):
            lo, hi = int(np.floor(tau)), int(np.ceil(tau))
            lam = (tau - lo) / (hi - lo)
            true_tau = frame_at(tau)

            d  = rel_l2(pred_frac[:, j], true_tau)
            pl = (1 - lam) * pred_int[:, int_index[lo]] + lam * pred_int[:, int_index[hi]]
            il = rel_l2(pl, true_tau)
            pt = (1 - lam) * frame_at(lo) + lam * frame_at(hi)
            it = rel_l2(pt, true_tau)

            phase = round(tau - lo, 2)
            rows.append((tau, phase, d, il, it))
            wcsv.writerow([f'{tau:.2f}', f'{phase:.2f}', f'{d:.6f}', f'{il:.6f}', f'{it:.6f}'])
    print(f"wrote {per_path}")

    # sanity: integer-time errors (should match paper-style per-step numbers)
    int_errs = {tau: rel_l2(pred_int[:, i], frame_at(tau)) for tau, i in int_index.items()}

    summ_path = os.path.join(args.save_dir, 'E4_summary.csv')
    with open(summ_path, 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['metric', 'lgn_direct', 'interp_lgn', 'interp_true'])
        arr = np.array([[r[2], r[3], r[4]] for r in rows])
        wcsv.writerow(['mean_all_frac'] + [f'{v:.6f}' for v in arr.mean(axis=0)])
        for ph in sorted(set(r[1] for r in rows)):
            sel = np.array([[r[2], r[3], r[4]] for r in rows if r[1] == ph])
            wcsv.writerow([f'mean_phase_{ph}'] + [f'{v:.6f}' for v in sel.mean(axis=0)])
        # early window (predictable regime) is where the claim is sharpest
        sel = np.array([[r[2], r[3], r[4]] for r in rows if r[0] <= 15.0])
        wcsv.writerow(['mean_frac_tau<=15'] + [f'{v:.6f}' for v in sel.mean(axis=0)])
    print(f"wrote {summ_path}")

    print("\n=== E4 summary ===")
    print(f"integer-time sanity: t=11 {int_errs[11]:.4f}, t=15 {int_errs.get(15, float('nan')):.4f}, "
          f"t=20 {int_errs.get(20, float('nan')):.4f}")
    print(f"fractional times, mean rel-L2 over {len(rows)} queries:")
    print(f"  LGN direct exp(L*t):    {arr.mean(axis=0)[0]:.4f}")
    print(f"  interp of LGN frames:   {arr.mean(axis=0)[1]:.4f}")
    print(f"  interp of TRUE frames:  {arr.mean(axis=0)[2]:.4f}  (oracle floor)")
    sel = np.array([[r[2], r[3], r[4]] for r in rows if r[0] <= 15.0])
    print(f"predictable window (tau <= 15):")
    print(f"  LGN direct / interp-LGN / interp-true: "
          f"{sel.mean(axis=0)[0]:.4f} / {sel.mean(axis=0)[1]:.4f} / {sel.mean(axis=0)[2]:.4f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data',         required=True)
    p.add_argument('--koopman_ckpt', required=True)
    p.add_argument('--save_dir',     default='./checkpoints')
    args = p.parse_args()
    main(args)
