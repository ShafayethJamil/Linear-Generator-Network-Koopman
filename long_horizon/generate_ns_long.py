"""Generate long-horizon 2D Navier-Stokes ground truth (nu = 1e-5, t = 1..200).

Paper: Appendix A.9, Table A.6 (data); run through long_horizon/run_pipeline.sh.

Usage (the paper's 200 trajectories are 4 x 50, seeds 1234-1237):
    python long_horizon/generate_ns_long.py --N 50 --T 200 --seed 1234 --out outputs/long_horizon/parts/long_s1234.mat

Same forcing and Gaussian-random-field initial conditions as the Li et al. benchmark, solved
at 256^2 with integrating-factor RK4 (CFL-adaptive, dt <= 1e-3) and subsampled to 64^2.
Output: HDF5 'u' of shape (N, 64, 64, T), frames at t = 1..T, plus the time vector 't'.
"""

import torch
import numpy as np
import math, argparse, os
from timeit import default_timer

torch.manual_seed(1234)
np.random.seed(1234)


# ============================================================
# Gaussian random field IC (FNO random_fields.py, 2D)
# ============================================================

class GaussianRF:
    def __init__(self, size, alpha=2.5, tau=7.0, device='cpu', dtype=torch.float32):
        self.size = size
        self.device = device
        self.dtype = dtype
        sigma = tau ** (0.5 * (2 * alpha - 2.0))          # dim = 2

        k_max = size // 2
        k = torch.cat([torch.arange(0, k_max), torch.arange(-k_max, 0)], dim=0)
        k1 = k.repeat(size, 1).transpose(0, 1)
        k2 = k.repeat(size, 1)

        sqrt_eig = (size ** 2) * math.sqrt(2.0) * sigma * \
                   ((4 * (math.pi ** 2) * (k1 ** 2 + k2 ** 2) + tau ** 2) ** (-alpha / 2.0))
        sqrt_eig[0, 0] = 0.0
        self.sqrt_eig = sqrt_eig.to(device)

    def sample(self, N):
        # Li et al. convention: real and imaginary parts each N(0,1) (complex
        # variance 2); a variance-1 complex draw gives half the benchmark energy.
        coeff = torch.view_as_complex(
            torch.randn(N, self.size, self.size, 2, device=self.device))
        coeff = self.sqrt_eig * coeff
        return torch.fft.ifft2(coeff).real.to(self.dtype)


# ============================================================
# 2D NS vorticity solver (FNO navier_stokes_2d, batched)
# ============================================================

def navier_stokes_2d(w0, f, visc, T, dt_max=1e-3, cfl=1.0, record_every=1.0):
    """
    w0: (N, s, s) initial vorticity. f: (s, s) forcing.
    Integrating-factor RK4 (exact viscous exponential + RK4 advection) with
    CFL-adaptive time step. Same physics as the FNO reference scheme; the
    time integrator is upgraded because explicit-Euler advection (used in the
    reference code, which only ran t<=20 at this viscosity) loses stability
    once the forced flow intensifies at long horizons.
    Records at t = record_every, 2*record_every, ..., T.
    Returns (N, s, s, n_records) on CPU float32.
    """
    device = w0.device
    N, s, _ = w0.shape

    w_h = torch.fft.fft2(w0.to(torch.float64))
    f_h = torch.fft.fft2(f.to(torch.float64))

    k_max = s // 2
    k = torch.cat([torch.arange(0, k_max, device=device),
                   torch.arange(-k_max, 0, device=device)], dim=0)
    k1 = k.repeat(s, 1).transpose(0, 1).to(torch.float64)
    k2 = k.repeat(s, 1).to(torch.float64)

    lap = 4 * (math.pi ** 2) * (k1 ** 2 + k2 ** 2)
    lap_ = lap.clone()
    lap_[0, 0] = 1.0

    dealias = ((torch.abs(k1) <= (2.0/3.0) * k_max) &
               (torch.abs(k2) <= (2.0/3.0) * k_max)).to(torch.float64)
    k_dealias = (2.0/3.0) * k_max                            # max retained |k| index

    def nonlin(wh):
        """N(w) = -(u.grad)w + f in spectral space, dealiased. Returns (F_h, umax)."""
        psi_h = wh / lap_
        q  = torch.fft.ifft2( 2 * math.pi * 1j * k2 * psi_h).real   # u
        v  = torch.fft.ifft2(-2 * math.pi * 1j * k1 * psi_h).real   # v
        wx = torch.fft.ifft2( 2 * math.pi * 1j * k1 * wh).real
        wy = torch.fft.ifft2( 2 * math.pi * 1j * k2 * wh).real
        Fh = -torch.fft.fft2(q * wx + v * wy) * dealias + f_h
        umax = max(q.abs().max().item(), v.abs().max().item())
        return Fh, umax

    records = []
    t = 0.0
    next_rec = record_every
    t0 = default_timer()
    n_steps = 0

    while t < T - 1e-12:
        F1, umax = nonlin(w_h)

        # CFL-adaptive dt: RK4 imaginary-axis limit ~2.83 / (2*pi*k_dealias*umax)
        dt = min(dt_max, cfl * 2.83 / (2 * math.pi * k_dealias * max(umax, 1e-8)))
        dt = min(dt, next_rec - t)

        E_half = torch.exp(-0.5 * dt * visc * lap)
        E_full = E_half * E_half

        # integrating-factor RK4 (Trefethen-style)
        u1 = E_half * (w_h + 0.5 * dt * F1)
        F2, _ = nonlin(u1)
        u2 = E_half * w_h + 0.5 * dt * F2
        F3, _ = nonlin(u2)
        u3 = E_full * w_h + dt * E_half * F3
        F4, _ = nonlin(u3)

        w_h = E_full * w_h + (dt / 6.0) * (E_full * F1 + 2 * E_half * (F2 + F3) + F4)
        t += dt
        n_steps += 1

        if t >= next_rec - 1e-12:
            w = torch.fft.ifft2(w_h).real
            if torch.isnan(w).any():
                raise RuntimeError(f"solver NaN at t={t:.2f}")
            records.append(w.cpu().to(torch.float32))
            if int(round(next_rec)) % 10 == 0:
                wall = default_timer() - t0
                print(f"  t={next_rec:6.1f}/{T}  max|w|={w.abs().max().item():.3f}  "
                      f"dt={dt:.2e}  steps={n_steps}  ({wall:.0f}s)", flush=True)
            next_rec += record_every

    return torch.stack(records, dim=-1)                     # (N, s, s, n_rec)


def main(args):
    if args.seed != 1234:                                   # 1234 already set at import
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"NS long-horizon truth: nu={args.visc}, N={args.N}, T={args.T}, "
          f"res {args.res}->{args.res_out}, dt_max={args.dt_max} (adaptive)")

    # forcing on the SOLVER grid
    x = torch.linspace(0, 1, args.res + 1, device=device)[:-1]
    X, Y = torch.meshgrid(x, x, indexing='ij')
    f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

    grf = GaussianRF(args.res, alpha=2.5, tau=7.0, device=device)

    sub = args.res // args.res_out
    outs = []
    done = 0
    while done < args.N:
        b = min(args.batch, args.N - done)
        print(f"Batch of {b} trajectories ({done}/{args.N} done):", flush=True)
        w0 = grf.sample(b)
        traj = navier_stokes_2d(w0, f, args.visc, args.T,
                                dt_max=args.dt_max, cfl=args.cfl, record_every=1.0)
        outs.append(traj[:, ::sub, ::sub, :])               # subsample like the paper
        done += b

    data = torch.cat(outs, dim=0).numpy().astype(np.float32)  # (N, 64, 64, T)
    print(f"\nOutput shape: {data.shape} | range [{data.min():.3f}, {data.max():.3f}] "
          f"mean {data.mean():.4f} std {data.std():.4f}")

    import h5py
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with h5py.File(args.out, 'w') as fo:
        fo.create_dataset('u', data=data, compression='gzip')
        fo.create_dataset('t', data=np.arange(1, args.T + 1, dtype=np.float32))
    print(f"Saved to {args.out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--N',        type=int,   default=50)
    p.add_argument('--T',        type=int,   default=200)
    p.add_argument('--visc',     type=float, default=1e-5)
    p.add_argument('--res',      type=int,   default=256)
    p.add_argument('--res_out',  type=int,   default=64)
    p.add_argument('--dt_max',   type=float, default=1e-3,
                   help='upper bound on adaptive step (CFL shrinks it as flow intensifies)')
    p.add_argument('--cfl',      type=float, default=1.0,
                   help='safety factor on the RK4 advective stability limit')
    p.add_argument('--batch',    type=int,   default=50,
                   help='trajectories solved simultaneously (memory permitting)')
    p.add_argument('--out',      default='outputs/long_horizon/ns_long_nu1e5_N50_T200.mat')
    p.add_argument('--seed',     type=int,   default=1234)
    args = p.parse_args()
    main(args)
