"""
2D FitzHugh-Nagumo reaction-diffusion data, periodic BCs (Appendix A.4)
========================================================================
Pseudo-spectral solver on [0, 2pi]^2 with RK4 time integration (dt = 0.01,
2/3 dealiasing of u^3). Only the activator u is stored (N, H, W, T+1); the
inhibitor v is hidden from the model.

    du/dt = D_u lap(u) + u - u^3/3 - v
    dv/dt = D_v lap(v) + eps (u + a - b v)

Paper datasets (D_v = 0.005, eps = 0.01, a = 0.7, b = 0.8, seed 42, 1200
trajectories, 64 x 64, 30 saved steps of 0.1 -> 31 frames, ~0.5 GB each):
    python fitzhugh_nagumo/generate_fhn_periodic.py --Du 0.01  --out data/fhn/rd_Du0.01_N1200_T30.mat
    python fitzhugh_nagumo/generate_fhn_periodic.py --Du 0.001 --out data/fhn/rd_Du0.001_N1200_T30.mat
"""

import numpy as np
import argparse, os
from timeit import default_timer


def generate_random_ic(res, rng=None):
    """
    Random initial condition for FitzHugh-Nagumo.
    u: small random perturbation around 0
    v: small random perturbation around 0
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.linspace(0, 2 * np.pi, res, endpoint=False)
    y = np.linspace(0, 2 * np.pi, res, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Random superposition of low-wavenumber modes
    u = np.zeros((res, res), dtype=np.float64)
    v = np.zeros((res, res), dtype=np.float64)

    n_modes = 10
    for _ in range(n_modes):
        kx = rng.integers(-4, 5)
        ky = rng.integers(-4, 5)
        if kx == 0 and ky == 0:
            continue
        # u perturbation
        amp_u = 0.3 * rng.uniform(0.3, 1.0)
        phase_u = rng.uniform(0, 2 * np.pi)
        u += amp_u * np.cos(kx * X + ky * Y + phase_u)

        # v perturbation (smaller amplitude)
        amp_v = 0.1 * rng.uniform(0.3, 1.0)
        phase_v = rng.uniform(0, 2 * np.pi)
        v += amp_v * np.cos(kx * X + ky * Y + phase_v)

    return u, v


class FitzHughNagumoSolver:
    """
    Pseudo-spectral 2D FitzHugh-Nagumo solver on [0, 2π]² with periodic BCs.
    """
    def __init__(self, res, Du=0.01, Dv=0.005, eps=0.01, a=0.7, b=0.8):
        self.res = res
        self.Du = Du
        self.Dv = Dv
        self.eps = eps
        self.a = a
        self.b = b

        # Wavenumbers
        k = np.fft.fftfreq(res, d=1.0/res)
        kx, ky = np.meshgrid(k, k, indexing='ij')
        self.k2 = kx**2 + ky**2

        # Dealiasing mask (2/3 rule)
        kmax = res // 3
        self.dealias = np.ones((res, res), dtype=np.float64)
        self.dealias[np.abs(kx) > kmax] = 0.0
        self.dealias[np.abs(ky) > kmax] = 0.0

    def rhs(self, u, v):
        """Compute RHS of FitzHugh-Nagumo."""
        # Diffusion in spectral space
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        diff_u = np.real(np.fft.ifft2(-self.Du * self.k2 * u_hat))
        diff_v = np.real(np.fft.ifft2(-self.Dv * self.k2 * v_hat))

        # Reaction (dealiased for cubic term)
        u3_hat = np.fft.fft2(u**3) * self.dealias
        u3 = np.real(np.fft.ifft2(u3_hat))

        # du/dt = D_u ∇²u + u - u³/3 - v
        du_dt = diff_u + u - u3/3.0 - v

        # dv/dt = D_v ∇²v + ε(u + a - b*v)
        dv_dt = diff_v + self.eps * (u + self.a - self.b * v)

        return du_dt, dv_dt

    def step_rk4(self, u, v, dt):
        """One RK4 step."""
        k1_u, k1_v = self.rhs(u, v)
        k2_u, k2_v = self.rhs(u + 0.5*dt*k1_u, v + 0.5*dt*k1_v)
        k3_u, k3_v = self.rhs(u + 0.5*dt*k2_u, v + 0.5*dt*k2_v)
        k4_u, k4_v = self.rhs(u + dt*k3_u, v + dt*k3_v)

        u_new = u + (dt/6) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
        v_new = v + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        return u_new, v_new

    def solve(self, u0, v0, dt, n_steps, save_every=1):
        """
        Integrate forward and save u snapshots only.
        Returns: u_snapshots of shape (T, res, res)
        """
        u = u0.copy()
        v = v0.copy()

        snapshots = [u.copy()]

        for step in range(1, n_steps + 1):
            u, v = self.step_rk4(u, v, dt)

            if np.any(np.isnan(u)) or np.any(np.abs(u) > 100):
                print(f"  WARNING: instability at step {step}, aborting")
                return None

            if step % save_every == 0:
                snapshots.append(u.copy())

        return np.stack(snapshots, axis=0)


def main(args):
    solver = FitzHughNagumoSolver(
        res=args.res, Du=args.Du, Dv=args.Dv,
        eps=args.eps, a=args.a, b=args.b)

    n_steps = args.T * args.save_every

    print(f"Reaction-Diffusion (FitzHugh-Nagumo) Data Generation")
    print(f"  Resolution: {args.res}x{args.res}")
    print(f"  Du={args.Du}, Dv={args.Dv}, eps={args.eps}, a={args.a}, b={args.b}")
    print(f"  dt={args.dt}, n_steps={n_steps}, save_every={args.save_every}")
    print(f"  Saved frames per trajectory: {args.T + 1}")
    print(f"  N trajectories: {args.N}")
    print(f"  Physics: dissipation scaling Re(lambda) ~ Du * |k|^2")
    print()

    rng = np.random.default_rng(seed=args.seed)

    all_data = []
    t0 = default_timer()
    good = 0
    attempt = 0

    while good < args.N:
        attempt += 1
        u0, v0 = generate_random_ic(args.res, rng=rng)

        result = solver.solve(u0, v0, args.dt, n_steps, args.save_every)

        if result is None:
            print(f"  Attempt {attempt}: failed, retrying...")
            continue

        # result shape: (T+1, res, res) -> (res, res, T+1) for NS convention
        all_data.append(result.transpose(1, 2, 0).astype(np.float32))
        good += 1

        if good % 100 == 0:
            elapsed = default_timer() - t0
            print(f"  Generated {good}/{args.N} trajectories ({elapsed:.1f}s)")

    elapsed = default_timer() - t0
    print(f"\nGenerated {args.N} trajectories in {elapsed:.1f}s")

    data = np.stack(all_data, axis=0)  # (N, H, W, T+1)
    print(f"Output shape: {data.shape}")
    print(f"  u range: [{data.min():.4f}, {data.max():.4f}]")
    print(f"  u mean:  {data.mean():.4f}")
    print(f"  u std:   {data.std():.4f}")

    import h5py
    with h5py.File(args.out, 'w') as f:
        f.create_dataset('u', data=data, compression='gzip')

    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N',         type=int,   default=1200)
    parser.add_argument('--res',       type=int,   default=64)
    parser.add_argument('--T',         type=int,   default=30)
    parser.add_argument('--dt',        type=float, default=0.01)
    parser.add_argument('--save_every',type=int,   default=10)
    parser.add_argument('--Du',        type=float, default=0.01,  help='Activator diffusion')
    parser.add_argument('--Dv',        type=float, default=0.005, help='Inhibitor diffusion')
    parser.add_argument('--eps',       type=float, default=0.01,  help='Timescale separation')
    parser.add_argument('--a',         type=float, default=0.7)
    parser.add_argument('--b',         type=float, default=0.8)
    parser.add_argument('--amplitude', type=float, default=0.3)
    parser.add_argument('--seed',      type=int,   default=42)
    parser.add_argument('--out',       default='rd_N1200_T30.mat')
    args = parser.parse_args()
    main(args)
