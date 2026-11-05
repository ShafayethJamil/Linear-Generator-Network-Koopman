"""Koopman neural operator (KNO) baseline, using the authors' koopmanlab package.

Paper: Table 1 (KNO row) and the KNO rows of Table A.6.

Usage (seeds 0, 1, 2):
    python baselines/kno.py --data data/ns/NavierStokes_V1e-5_N1200_T20.mat --tag kno_nu1e5_s0 --seed 0
    python baselines/kno.py --data data/ns/ns_V1e-3_N5000_T50.mat --T_out 40 --tag kno_nu1e3_s0 --seed 0

Requires koopmanlab (pip install koopmanlab; 1.0.3 was used). Same data split and T_in/T_out
as LGN-KM; the test error is computed with the paper metric, not the package's own.
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse, os, csv

def set_seed(seed):
    """Seed torch (weights, KNO training) and numpy. KNO is stochastic, so
    seeds give genuine variance (unlike the deterministic DMD baseline)."""
    torch.manual_seed(seed)
    np.random.seed(seed)


# ============================================================
# Data loading — identical protocol (RAW, no normalization)
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
        u = torch.from_numpy(arr).permute(3, 1, 2, 0)
    else:
        u = torch.from_numpy(arr)

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

    x_train = u[:n_train,               :, :, :T_in]
    y_train = u[:n_train,               :, :, T_in:T_in+T_out]
    x_test  = u[n_train:n_train+n_test, :, :, :T_in]
    y_test  = u[n_train:n_train+n_test, :, :, T_in:T_in+T_out]
    print(f"  Train: {tuple(x_train.shape)} -> {tuple(y_train.shape)}  (RAW, no normalization)")
    return x_train, y_train, x_test, y_test, H, W


def paper_rel_l2(pred, target):
    """(B, ...) -> mean over B of ||pred-target|| / ||target|| on flattened volume."""
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = target.reshape(B, -1)
    return float(((p - t).norm(dim=1) / (t.norm(dim=1) + 1e-8)).mean())


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    try:
        import koopmanlab as kp
    except ImportError:
        raise SystemExit("koopmanlab not installed. Run: pip install koopmanlab "
                         "(or clone github.com/Koopman-Laboratory/KoopmanLab and pip install -e .)")

    # koopmanlab/models/kno.py does `torch.manual_seed(0)` AT IMPORT, which
    # silently overrides our set_seed(args.seed) call. Re-apply the seed HERE,
    # after the import, so the model's torch.rand init actually varies per seed.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_train, y_train, x_test, y_test, H, W = load_data(
        args.data, args.T_in, args.T_out, args.n_train, args.n_test)

    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    test_ds  = torch.utils.data.TensorDataset(x_test,  y_test)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    eval_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    # ---------------- BUILD (marked call site #1) ----------------
    # KNO2d per the authors' README/examples:
    #   o = operator (lifting) dimension, m = Fourier modes kept,
    #   r = power of the Koopman operator applied per step, t_in = input frames.
    KNO_model = kp.model.koopman(backbone="KNO2d", autoencoder="MLP",
                                 o=args.o, m=args.m, r=args.r_power,
                                 t_in=args.T_in, device=device)
    KNO_model.compile()
    KNO_model.opt_init("Adam", lr=args.lr, step_size=100, gamma=0.5)

    # ---------------- TRAIN (marked call site #2) ----------------
    # koopmanlab 1.0.3: the multi-step trainer is `train` (there is NO
    # train_multi). step=1 slides the input window by one frame per predicted
    # step; T_out matches our rollout horizon. Verified against the installed
    # package API (koopman.train signature: epochs, trainloader, step, T_out,
    # evalloader).
    KNO_model.train(epochs=args.epochs,
                    trainloader=train_loader,
                    step=1, T_out=args.T_out,
                    evalloader=eval_loader)

    # ---------------- PROTOCOL-IDENTICAL EVALUATION ----------------
    # We compute OUR paper metric on the trained network directly.
    # koopmanlab stores the underlying nn.Module; the attribute name has
    # varied across versions — try the known ones.
    net = None
    for attr in ('kernel', 'kno_model', 'model', 'net'):
        if hasattr(KNO_model, attr) and isinstance(getattr(KNO_model, attr), torch.nn.Module):
            net = getattr(KNO_model, attr)
            break
    if net is None:
        # In koopmanlab 1.0.3 the module is always KNO_model.kernel, so this
        # branch should be unreachable. If the API changes, fail loudly rather
        # than silently reporting the package's own (different) metric.
        raise SystemExit("Could not locate koopmanlab's underlying nn.Module "
                         "(expected KNO_model.kernel). Inspect dir(KNO_model) "
                         "before proceeding.")

    net.eval()
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- persist the trained model so KNO can be re-evaluated / extended
    # (long-horizon, alt metrics) without retraining. Save both the underlying
    # nn.Module state_dict and the build config needed to reconstruct it. ----
    ckpt_path = os.path.join(args.save_dir, f'best_kno_{args.tag}.pt')
    torch.save({'state_dict': net.state_dict(),
                'config': {'backbone': 'KNO2d', 'autoencoder': 'MLP',
                           'o': args.o, 'm': args.m, 'r_power': args.r_power,
                           'T_in': args.T_in, 'T_out': args.T_out,
                           'epochs': args.epochs}},
               ckpt_path)
    print(f"saved KNO checkpoint -> {ckpt_path}")

    per_step_acc = np.zeros(args.T_out)
    total_preds, total_targs = [], []
    with torch.no_grad():
        for xb, yb in eval_loader:
            xb = xb.to(device)                       # (B,H,W,T_in)
            # KNO2d predicts the next frame(s) from the input window and is
            # rolled out autoregressively for multi-step prediction.
            cur = xb
            preds = []
            for _ in range(args.T_out):
                # KNO2d.forward returns (prediction, reconstruction); the last
                # channel of the prediction is the next-step frame (matches the
                # package's own rollout in koopman.train).
                im, _ = net(cur)                     # (B,H,W,T_in), (B,H,W,T_in)
                out = im[..., -1:]                   # (B,H,W,1)
                preds.append(out)
                cur = torch.cat([cur[..., 1:], out], dim=-1)  # slide window (width T_in)
            pred = torch.cat(preds, dim=-1).cpu()    # (B,H,W,T_out)
            total_preds.append(pred.permute(0, 3, 1, 2))
            total_targs.append(yb.permute(0, 3, 1, 2))

    pred = torch.cat(total_preds)                    # (B,T_out,H,W)
    targ = torch.cat(total_targs)
    total = paper_rel_l2(pred, targ)
    per_step = [paper_rel_l2(pred[:, t], targ[:, t]) for t in range(args.T_out)]

    out_path = os.path.join(args.save_dir, f'kno_results_{args.tag}.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['test_rel_l2_full_volume', f'{total:.6f}'])
        for t, v in enumerate(per_step, 1):
            w.writerow([f'l2_step{t}', f'{v:.6f}'])
        w.writerow(['o', args.o]); w.writerow(['m', args.m])
        w.writerow(['r_power', args.r_power]); w.writerow(['epochs', args.epochs])

    print(f"\n{'='*60}\nKNO [{args.tag}] test rel-L2 (paper metric) = {total:.4f}")
    print(f"per-step: {['%.3f' % v for v in per_step]}")
    print(f"wrote {out_path}\n{'='*60}")
    print("Table note: KNO exposes NO structured continuous-time generator — "
          "no S-D split, no k-resolved eigenspectrum, no dispersion readout.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     required=True)
    parser.add_argument('--tag',      default='kno_nu1e5')
    parser.add_argument('--seed',     type=int,   default=0,
                        help='random seed (include it in --tag, e.g. _s1)')
    parser.add_argument('--T_in',     type=int,   default=10)
    parser.add_argument('--T_out',    type=int,   default=10)
    parser.add_argument('--n_train',  type=int,   default=1000)
    parser.add_argument('--n_test',   type=int,   default=200)
    parser.add_argument('--o',        type=int,   default=32,
                        help='KNO operator/lifting dimension (paper NS default)')
    parser.add_argument('--m',        type=int,   default=16,
                        help='KNO Fourier modes kept')
    parser.add_argument('--r_power',  type=int,   default=8,
                        help='KNO Koopman-operator power per step')
    parser.add_argument('--epochs',   type=int,   default=500)
    parser.add_argument('--batch',    type=int,   default=10)
    parser.add_argument('--lr',       type=float, default=1e-3)
    parser.add_argument('--save_dir', default='outputs/baselines')
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
