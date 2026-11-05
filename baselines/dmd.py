"""Exact DMD baseline on the raw Navier-Stokes field (CPU, numpy).

Paper: Table 1 (DMD row: test relative L2, max|mu|).

Usage:
    python baselines/dmd.py --data data/ns/NavierStokes_V1e-5_N1200_T20.mat --tag dmd_nu1e5
    python baselines/dmd.py --data data/ns/ns_V1e-3_N5000_T50.mat --T_out 40 --tag dmd_nu1e3

Snapshot pairs are pooled over the training trajectories; each test trajectory is rolled
forward from its last input frame. Table 1 reports the best of ranks 32-256.
Writes dmd_results_<tag>.csv and dmd_eigs_<tag>.csv to --save_dir.
"""

import numpy as np
import argparse, os, csv


# ============================================================
# Data loading — mirrors the torch loader (NO normalization)
# ============================================================

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

    if arr.ndim != 4:
        raise ValueError(f"Unexpected shape: {arr.shape}")
    if arr.shape[0] < arr.shape[-1]:
        u = np.transpose(arr, (3, 1, 2, 0))   # -> (N,H,W,T)
    else:
        u = arr

    N, H, W, T = u.shape
    print(f"  Parsed: N={N} H={H} W={W} T={T}")

    if H != target_res or W != target_res:
        # bilinear downsample via simple block averaging fallback
        try:
            from scipy.ndimage import zoom
            factor = target_res / H
            u = zoom(u, (1, factor, factor, 1), order=1)
        except Exception:
            step = H // target_res
            u = u[:, ::step, ::step, :]
        H = W = target_res
        print(f"  Downsampled to {H}x{W}")

    assert T >= T_in + T_out
    print(f"  Data stats: mean={u.mean():.4f} std={u.std():.4f} (NO normalization)")
    return u, N, H, W, T


def rel_l2(pred, target):
    """pred/target: (B, T_out*H*W). Per-sample relative L2, then mean."""
    num = np.linalg.norm(pred - target, axis=1)
    den = np.linalg.norm(target, axis=1) + 1e-8
    return float((num / den).mean())


def main(args):
    # Exact DMD is deterministic (fixed split, no shuffling, no random init),
    # so results are seed-invariant; --seed is wired only for CLI uniformity
    # with the neural baselines and to seed the zoom/downsample fallback path.
    np.random.seed(args.seed)
    u, N, H, W, T = load_data(args.data, args.T_in, args.T_out,
                              args.n_train, args.n_test)
    D = H * W
    train = u[:args.n_train]                                    # (n_train,H,W,T)
    test  = u[args.n_train:args.n_train + args.n_test]

    # ---- snapshot pair matrices from training trajectories ----
    # X: states at t, Y: states at t+1, pooled over all trajectories/frames
    Xl, Yl = [], []
    for n in range(train.shape[0]):
        traj = train[n].reshape(D, T)                           # (D, T)
        Xl.append(traj[:, :-1])
        Yl.append(traj[:, 1:])
    X = np.concatenate(Xl, axis=1).astype(np.float64)           # (D, n_train*(T-1))
    Y = np.concatenate(Yl, axis=1).astype(np.float64)
    print(f"Snapshot matrices: X {X.shape}, Y {Y.shape}")

    # ---- economy SVD of X (compute once; reuse across ranks) ----
    print("Computing SVD of X (this is the slow step) ...")
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    print(f"  SVD done. Leading sigma: {s[:5].round(3)} ... "
          f"energy in top {max(args.ranks)}: "
          f"{(s[:max(args.ranks)]**2).sum()/(s**2).sum()*100:.2f}%")

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f'dmd_results_{args.tag}.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'test_rel_l2', 'pct_stable_eigs', 'max_abs_mu']
                   + [f'l2_step{t}' for t in range(1, args.T_out + 1)])

    # full DMD eigenvalue spectrum per rank (%stable and max|mu|). mu is discrete-time; log(mu) gives the
    # continuous-time generator eigenvalue for comparison with LGN's Re(lambda).
    eig_path = os.path.join(args.save_dir, f'dmd_eigs_{args.tag}.csv')
    with open(eig_path, 'w', newline='') as f:
        csv.writer(f).writerow(['rank', 're_mu', 'im_mu', 'abs_mu',
                                're_log_mu', 'im_log_mu'])

    for rank in args.ranks:
        Ur  = U[:, :rank]
        sr  = s[:rank]
        Vr  = Vt[:rank, :].T
        # exact DMD reduced operator: Atilde = Ur^T Y Vr Sigma^-1
        Atilde = Ur.T @ Y @ (Vr / sr)                           # (rank, rank)

        mu = np.linalg.eigvals(Atilde)
        pct_stable = 100.0 * float((np.abs(mu) <= 1.0 + 1e-9).mean())
        max_mu = float(np.abs(mu).max())

        with open(eig_path, 'a', newline='') as f:
            wri = csv.writer(f)
            for m in mu:
                lm = np.log(m) if m != 0 else complex(-np.inf, 0)
                wri.writerow([rank, f'{m.real:.6e}', f'{m.imag:.6e}',
                              f'{abs(m):.6e}', f'{lm.real:.6e}', f'{lm.imag:.6e}'])

        # ---- forecast held-out trajectories ----
        # condition on frame index (T_in - 1) = last input frame,
        # predict frames T_in .. T_in+T_out-1
        x0 = test[:, :, :, args.T_in - 1].reshape(-1, D).T.astype(np.float64)  # (D,B)
        z  = Ur.T @ x0                                          # (rank, B)
        preds = []
        for _ in range(args.T_out):
            z = Atilde @ z
            preds.append((Ur @ z).T)                            # (B, D)
        pred = np.stack(preds, axis=1)                          # (B, T_out, D)
        targ = np.transpose(
            test[:, :, :, args.T_in:args.T_in + args.T_out],
            (0, 3, 1, 2)).reshape(test.shape[0], args.T_out, D)

        total = rel_l2(pred.reshape(test.shape[0], -1),
                       targ.reshape(test.shape[0], -1))
        per_step = [rel_l2(pred[:, t].reshape(test.shape[0], -1),
                           targ[:, t].reshape(test.shape[0], -1))
                    for t in range(args.T_out)]

        with open(out_path, 'a', newline='') as f:
            csv.writer(f).writerow(
                [rank, f'{total:.6f}', f'{pct_stable:.2f}', f'{max_mu:.6f}']
                + [f'{v:.6f}' for v in per_step])

        print(f"[rank {rank:4d}] test rel-L2 = {total:.4f} | "
              f"stable eigs = {pct_stable:.1f}% | max|mu| = {max_mu:.4f} | "
              f"step1 {per_step[0]:.4f} -> step{args.T_out} {per_step[-1]:.4f}")

    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    required=True)
    parser.add_argument('--tag',     default='dmd_nu1e5')
    parser.add_argument('--seed',    type=int, default=0,
                        help='seed (DMD is deterministic; kept for CLI uniformity)')
    parser.add_argument('--T_in',    type=int, default=10)
    parser.add_argument('--T_out',   type=int, default=10)
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_test',  type=int, default=200)
    parser.add_argument('--ranks',   type=int, nargs='+',
                        default=[32, 64, 128, 256])
    parser.add_argument('--save_dir', default='outputs/baselines')
    args = parser.parse_args()
    main(args)
