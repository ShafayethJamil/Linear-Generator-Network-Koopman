"""Generate Navier-Stokes ground truth sampled every 0.25 time units (nu = 1e-5, t = 0.25..20).

Paper: Section 4.1 and Appendix A.9, Table A.7 (queries between training frames); run through
long_horizon/run_pipeline.sh.

Usage (the paper's 200 trajectories are 4 x 50, seeds 4321-4324):
    python long_horizon/generate_ns_finedt.py --N 50 --seed 4321 --out outputs/long_horizon/parts/finedt_s4321.mat

Same solver and initial conditions as generate_ns_long.py, at half its CFL number (0.5) so the
reference error sits well below the sub-frame accuracy being measured.
"""

import torch
import numpy as np
import math, argparse, os
from generate_ns_long import GaussianRF, navier_stokes_2d

torch.manual_seed(4321)          # different seed than generate_ns_long.py -> fresh ICs
np.random.seed(4321)


def main(args):
    if args.seed != 4321:                                   # 4321 already set at import
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"NS fine-dt truth: nu={args.visc}, N={args.N}, T={args.T}, "
          f"record_every={args.record_every}, res {args.res}->{args.res_out}")

    x = torch.linspace(0, 1, args.res + 1, device=device)[:-1]
    X, Y = torch.meshgrid(x, x, indexing='ij')
    f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

    grf = GaussianRF(args.res, alpha=2.5, tau=7.0, device=device)
    sub = args.res // args.res_out

    outs, done = [], 0
    while done < args.N:
        b = min(args.batch, args.N - done)
        print(f"Batch of {b} trajectories ({done}/{args.N} done):", flush=True)
        w0 = grf.sample(b)
        traj = navier_stokes_2d(w0, f, args.visc, args.T,
                                dt_max=args.dt_max, cfl=args.cfl,
                                record_every=args.record_every)
        outs.append(traj[:, ::sub, ::sub, :])
        done += b

    data = torch.cat(outs, dim=0).numpy().astype(np.float32)
    t_rec = np.arange(1, data.shape[-1] + 1, dtype=np.float32) * args.record_every
    print(f"\nOutput shape: {data.shape} | frames at t = {t_rec[0]:.2f}..{t_rec[-1]:.2f} "
          f"step {args.record_every}")
    print(f"range [{data.min():.3f}, {data.max():.3f}] mean {data.mean():.4f} std {data.std():.4f}")

    import h5py
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with h5py.File(args.out, 'w') as fo:
        fo.create_dataset('u', data=data, compression='gzip')
        fo.create_dataset('t', data=t_rec)
    print(f"Saved to {args.out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--N',            type=int,   default=50)
    p.add_argument('--T',            type=float, default=20.0)
    p.add_argument('--visc',         type=float, default=1e-5)
    p.add_argument('--res',          type=int,   default=256)
    p.add_argument('--res_out',      type=int,   default=64)
    p.add_argument('--record_every', type=float, default=0.25)
    p.add_argument('--dt_max',       type=float, default=1e-3)
    p.add_argument('--cfl',          type=float, default=0.5,
                   help='half the RK4 stability edge — ground-truth accuracy margin')
    p.add_argument('--batch',        type=int,   default=50)
    p.add_argument('--out',          default='outputs/long_horizon/ns_finedt_nu1e5_N50_T20.mat')
    p.add_argument('--seed',         type=int,   default=4321)
    args = p.parse_args()
    main(args)
