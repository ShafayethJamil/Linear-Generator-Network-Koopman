"""
2D FitzHugh-Nagumo with no-flux (Neumann) BCs (Appendix A.7)
=============================================================
Same physics, parameters, protocol and output format as generate_fhn_periodic.py.
Only change: homogeneous Neumann boundaries on [0, 2pi]^2, solved
pseudo-spectrally in the cosine basis (DCT-II diagonalizes the Laplacian under
Neumann BCs), RK4 in time. Cosine wavenumbers k_m = pi m / L. Initial
conditions are sums of cos(m pi x/L) cos(n pi y/L) (zero normal derivative).

Paper datasets (seed 42, 1200 trajectories, 64 x 64, 31 frames):
    python fitzhugh_nagumo/generate_fhn_neumann.py --Du 0.01  --out data/fhn/rd_neumann_Du0.01_N1200_T30.mat
    python fitzhugh_nagumo/generate_fhn_neumann.py --Du 0.001 --out data/fhn/rd_neumann_Du0.001_N1200_T30.mat
"""

import numpy as np
import argparse
from timeit import default_timer
from scipy.fft import dctn, idctn


def generate_random_ic(res, L, rng=None, n_modes=10):
    """Neumann-compatible random IC: sum of cos(m pi x/L) cos(n pi y/L)."""
    if rng is None:
        rng = np.random.default_rng()

    # cell-centered grid (matches DCT-II convention)
    h = L / res
    x = (np.arange(res) + 0.5) * h
    X, Y = np.meshgrid(x, x, indexing='ij')

    u = np.zeros((res, res), dtype=np.float64)
    v = np.zeros((res, res), dtype=np.float64)

    for _ in range(n_modes):
        m = rng.integers(0, 5)
        n = rng.integers(0, 5)
        if m == 0 and n == 0:
            continue
        cmn = np.cos(m * np.pi * X / L) * np.cos(n * np.pi * Y / L)
        u += 0.3 * rng.uniform(0.3, 1.0) * rng.choice([-1, 1]) * cmn
        v += 0.1 * rng.uniform(0.3, 1.0) * rng.choice([-1, 1]) * cmn

    return u, v


class FHNNeumannSolver:
    """Pseudo-spectral (cosine basis) 2D FitzHugh-Nagumo on [0,L]^2, no-flux BCs."""

    def __init__(self, res, L=2*np.pi, Du=0.01, Dv=0.005, eps=0.01, a=0.7, b=0.8):
        self.res, self.L = res, L
        self.Du, self.Dv, self.eps, self.a, self.b = Du, Dv, eps, a, b

        # cosine wavenumbers k_m = pi*m/L
        m = np.arange(res)
        km = np.pi * m / L
        kx, ky = np.meshgrid(km, km, indexing='ij')
        self.k2 = kx**2 + ky**2

        # 2/3-rule mask on cosine coefficients (stabilizes the cubic term)
        mmax = res * 2 // 3
        self.dealias = np.ones((res, res), dtype=np.float64)
        self.dealias[mmax:, :] = 0.0
        self.dealias[:, mmax:] = 0.0

    def _lap(self, f, Dcoef):
        f_hat = dctn(f, type=2, norm='ortho')
        return idctn(-Dcoef * self.k2 * f_hat, type=2, norm='ortho')

    def rhs(self, u, v):
        diff_u = self._lap(u, self.Du)
        diff_v = self._lap(v, self.Dv)

        u3_hat = dctn(u**3, type=2, norm='ortho') * self.dealias
        u3 = idctn(u3_hat, type=2, norm='ortho')

        du_dt = diff_u + u - u3/3.0 - v
        dv_dt = diff_v + self.eps * (u + self.a - self.b * v)
        return du_dt, dv_dt

    def step_rk4(self, u, v, dt):
        k1_u, k1_v = self.rhs(u, v)
        k2_u, k2_v = self.rhs(u + 0.5*dt*k1_u, v + 0.5*dt*k1_v)
        k3_u, k3_v = self.rhs(u + 0.5*dt*k2_u, v + 0.5*dt*k2_v)
        k4_u, k4_v = self.rhs(u + dt*k3_u, v + dt*k3_v)
        return (u + (dt/6)*(k1_u + 2*k2_u + 2*k3_u + k4_u),
                v + (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v))

    def solve(self, u0, v0, dt, n_steps, save_every=1):
        u, v = u0.copy(), v0.copy()
        snapshots = [u.copy()]
        for step in range(1, n_steps + 1):
            u, v = self.step_rk4(u, v, dt)
            if np.any(np.isnan(u)) or np.any(np.abs(u) > 100):
                print(f"  WARNING: instability at step {step}, aborting")
                return None
            if step % save_every == 0:
                snapshots.append(u.copy())
        return np.stack(snapshots, axis=0)

    def check_noflux(self, u):
        """Wall-normal derivative estimated from the grid values themselves: a
        second-order one-sided difference through the three cell centres nearest
        each wall, du/dn(wall) ~ (-2 u_0 + 3 u_1 - u_2) / h, at all four walls,
        compared with the largest centred-difference gradient in the interior.
        A no-flux field gives a ratio of O(h^2); a field that violates the
        boundary condition gives O(1). Returns (max |du/dn| at walls, ratio)."""
        # fix: the original multiplied by 0.0 and always returned 0 (no-op check)
        h = self.L / self.res
        one_sided = lambda a, b, c: np.abs(-2.0 * a + 3.0 * b - c) / h
        wall = max(one_sided(u[0], u[1], u[2]).max(), one_sided(u[-1], u[-2], u[-3]).max(),
                   one_sided(u[:, 0], u[:, 1], u[:, 2]).max(), one_sided(u[:, -1], u[:, -2], u[:, -3]).max())
        gx = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * h)
        gy = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * h)
        interior = np.sqrt(gx**2 + gy**2).max()
        return wall, wall / (interior + 1e-300)


def main(args):
    solver = FHNNeumannSolver(res=args.res, Du=args.Du, Dv=args.Dv,
                              eps=args.eps, a=args.a, b=args.b)
    n_steps = args.T * args.save_every

    print("FitzHugh-Nagumo, NO-FLUX (Neumann) BCs, cosine pseudo-spectral")
    print(f"  Resolution {args.res}x{args.res} | Du={args.Du} Dv={args.Dv} "
          f"eps={args.eps} a={args.a} b={args.b}")
    print(f"  dt={args.dt} n_steps={n_steps} save_every={args.save_every} "
          f"-> {args.T+1} frames x {args.N} trajectories")
    print(f"  Target physics: Re(lambda) ~ -Du*|k|^2, k_m = pi*m/L (cosine)")
    print()

    rng = np.random.default_rng(seed=args.seed)
    all_data, good, attempt = [], 0, 0
    t0 = default_timer()

    while good < args.N:
        attempt += 1
        u0, v0 = generate_random_ic(args.res, solver.L, rng=rng)
        result = solver.solve(u0, v0, args.dt, n_steps, args.save_every)
        if result is None:
            print(f"  Attempt {attempt}: failed, retrying...")
            continue
        all_data.append(result.transpose(1, 2, 0).astype(np.float32))
        good += 1
        if good == 1:
            wall, ratio = solver.check_noflux(result[-1].astype(np.float64))
            print(f"  no-flux BC check (final frame of trajectory 1): max |du/dn| at walls "
                  f"= {wall:.2e}, {ratio:.2e} of the interior max |grad u|")
        if good % 100 == 0:
            print(f"  Generated {good}/{args.N} ({default_timer()-t0:.1f}s)")

    data = np.stack(all_data, axis=0)   # (N, H, W, T+1)
    print(f"\nGenerated {args.N} trajectories in {default_timer()-t0:.1f}s")
    print(f"Output shape: {data.shape} | u range [{data.min():.4f}, {data.max():.4f}] "
          f"mean {data.mean():.4f} std {data.std():.4f}")

    import h5py
    with h5py.File(args.out, 'w') as f:
        f.create_dataset('u', data=data, compression='gzip')
    print(f"Saved to {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N',          type=int,   default=1200)
    parser.add_argument('--res',        type=int,   default=64)
    parser.add_argument('--T',          type=int,   default=30)
    parser.add_argument('--dt',         type=float, default=0.01)
    parser.add_argument('--save_every', type=int,   default=10)
    parser.add_argument('--Du',         type=float, default=0.01)
    parser.add_argument('--Dv',         type=float, default=0.005)
    parser.add_argument('--eps',        type=float, default=0.01)
    parser.add_argument('--a',          type=float, default=0.7)
    parser.add_argument('--b',          type=float, default=0.8)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--out',        default='rd_neumann_N1200_T30.mat')
    args = parser.parse_args()
    main(args)
