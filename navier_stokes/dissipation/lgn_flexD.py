"""Train LGN-KM with a relaxed dissipation form: a learnable exponent p or a free diagonal.

Paper: Section 3.2, Fig. 3a, Appendix A.6, Table A.3 (checkpoints/dissipation/).

Usage:
    python navier_stokes/dissipation/lgn_flexD.py --data data/ns/NavierStokes_V1e-5_N1200_T20.mat --Dparam freeexp --tag flexD_freeexp_nu1e5 --seed 0 --save_dir outputs/dissipation
    python navier_stokes/dissipation/lgn_flexD.py --data data/ns/ns_V1e-3_N5000_T50.mat --T_out 40 --Dparam freeexp --tag flexD_freeexp_nu1e3 --seed 0 --save_dir outputs/dissipation
        (--Dparam free for the free diagonal; seeds 0, 1, 2 with tags <tag>, <tag>_s1, <tag>_s2;
         add --analyze to write the eigenspectrum and exponent fit of a trained model)

--Dparam freeexp: D_k = d0 + a0 |k|^p with p learnable (init 1); free: D_k = softplus(B_k), no form in k;
paper: D_k = d0 + a0 |k|^2. S stays skew and D_k positive, so every eigenvalue has Re <= 0.
nu = 1e-3 uses --T_out 40 (with the default T_out 10, p ends near 1.2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import matrix_exp
import numpy as np
import argparse, os, csv, math
from timeit import default_timer


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def inv_softplus(y):
    """x such that softplus(x) = y (y > 0)."""
    return math.log(math.expm1(y))


# ============================================================
# Spectral Conv 2D (identical to paper code)
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
# Encoder / Decoder (identical to paper code)
# ============================================================

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
# FLEX-D MODEL
#   S = (P - P^T)/2                         (unchanged, shared across modes)
#   freeexp: D = softplus(d) + softplus(alpha) * knorm^softplus(p_raw)
#   free   : D = softplus(B),  B free (modes,modes,r)
#   paper  : D = softplus(d) + softplus(alpha) * knorm^2
#   L_k = S - diag(D_k)   ->  Re(lambda) <= 0 by construction in ALL variants
# ============================================================

class SpatialKoopmanFlexD(nn.Module):
    def __init__(self, modes, width, r, T_in, H, W, Dparam='freeexp', fix_p=None):
        super().__init__()
        self.modes, self.r, self.H, self.W = modes, r, H, W
        self.Dparam = Dparam
        self.fix_p = fix_p
        self.encoder = SpatialEncoder(modes, width, r, T_in)
        self.decoder = SpatialDecoder(r)
        self.P = nn.Parameter(torch.randn(r, r) * 0.01)

        k1 = torch.arange(modes, dtype=torch.float32)
        k2 = torch.arange(modes, dtype=torch.float32)
        kk1, kk2 = torch.meshgrid(k1, k2, indexing='ij')
        k2_norm = (kk1**2 + kk2**2) / (modes**2)
        self.register_buffer('k2_norm', k2_norm)                    # |k|^2 (normalized)
        self.register_buffer('knorm', torch.sqrt(k2_norm))          # |k|   (normalized)

        if Dparam in ('freeexp', 'paper'):
            self.d     = nn.Parameter(torch.full((r,), -2.0))
            self.alpha = nn.Parameter(torch.full((r,), -3.0))
            if Dparam == 'freeexp':
                if fix_p is not None:
                    # FIXED-EXPONENT CONTROL: p frozen at fix_p, everything
                    # else trains. Used for the loss-vs-p scan.
                    self.p_raw = nn.Parameter(torch.tensor(inv_softplus(fix_p)),
                                              requires_grad=False)
                else:
                    # softplus(0.5413) = 1.0 -> exponent initialized FAR from 2
                    self.p_raw = nn.Parameter(torch.tensor(0.5413))
        elif Dparam == 'free':
            # completely free positive diagonal per (k1,k2,latent-dim).
            # softplus(-2.0) ~ 0.127 matches the paper init scale of d0.
            self.B = nn.Parameter(torch.full((modes, modes, r), -2.0))
        else:
            raise ValueError(f"unknown Dparam {Dparam}")

    def get_exponent(self):
        if self.Dparam == 'freeexp':
            return F.softplus(self.p_raw).item()
        return float('nan')

    def get_D_diag(self):
        if self.Dparam == 'free':
            return F.softplus(self.B)                               # (modes,modes,r)
        d0 = F.softplus(self.d)
        a0 = F.softplus(self.alpha)
        if self.Dparam == 'paper':
            kterm = self.k2_norm
        else:  # freeexp
            p = F.softplus(self.p_raw)
            kterm = torch.pow(self.knorm + 1e-8, p)
        return d0.unsqueeze(0).unsqueeze(0) + \
               a0.unsqueeze(0).unsqueeze(0) * kterm.unsqueeze(-1)

    def get_generators(self):
        S = (self.P - self.P.T) / 2.0
        D_diag = self.get_D_diag()
        S_exp = S.unsqueeze(0).unsqueeze(0).expand(self.modes, self.modes, -1, -1).clone()
        D_mat = torch.diag_embed(D_diag)
        return S_exp - D_mat

    def propagate(self, z0_ft, t, dt=1.0):
        L_all = self.get_generators()
        eL = matrix_exp(L_all * t * dt)
        z = z0_ft.permute(0, 2, 3, 1).unsqueeze(-1)
        eL_exp = eL.unsqueeze(0)
        z_t = (eL_exp.to(torch.cfloat) @ z).squeeze(-1)
        return z_t.permute(0, 3, 1, 2)

    def forward(self, x, grid, T_out, dt=1.0):
        z0    = self.encoder(x, grid)
        z0_ft = torch.fft.rfft2(z0)
        z0_ft_trunc = z0_ft[:, :, :self.modes, :self.modes]

        preds = []
        for t in range(1, T_out + 1):
            zt_ft_trunc = self.propagate(z0_ft_trunc, t, dt)
            zt_ft = z0_ft.clone()
            zt_ft[:, :, :self.modes, :self.modes] = zt_ft_trunc
            z_t = torch.fft.irfft2(zt_ft, s=(self.H, self.W))
            preds.append(self.decoder(z_t))
        return torch.stack(preds, dim=1)

    def get_max_re_eigenvalue(self):
        L_all = self.get_generators()
        L_flat = L_all.reshape(-1, self.r, self.r)
        eigs = torch.linalg.eigvals(L_flat)
        return eigs.real.max().item()


# ============================================================
# Paper metric (identical)
# ============================================================

def paper_rel_l2_loss(pred, target):
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = target.reshape(B, -1)
    return ((p - t).norm(dim=1) / (t.norm(dim=1) + 1e-8)).mean()


# ============================================================
# Data loading (identical, NO normalization)
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
# POST-HOC ANALYSIS
#   For every (k1,k2): eigvals of L_k; dominant (max-Re) eigenvalue.
#   Power-law fit:  log(-Re_dom) = p_hat * log|k| + c   (shells k>0)
#   Quadratic fit:  -Re_dom = a*|k|^2 + b               (R2 reported)
# ============================================================

def analyze(model, tag, save_dir):
    model.eval()
    modes, r = model.modes, model.r
    with torch.no_grad():
        L_all = model.get_generators().cpu()           # (modes,modes,r,r)
        knorm = model.knorm.cpu().numpy()
        k2n   = model.k2_norm.cpu().numpy()

    rows, dom = [], []
    n_total, n_stable = 0, 0
    max_re_global = -np.inf
    for i in range(modes):
        for j in range(modes):
            eigs = torch.linalg.eigvals(L_all[i, j]).numpy()
            n_total += eigs.size
            n_stable += int((eigs.real <= 1e-9).sum())
            max_re_global = max(max_re_global, float(eigs.real.max()))
            for e in eigs:
                rows.append([e.real, e.imag, k2n[i, j]])
            dom.append([knorm[i, j], float(eigs.real.max())])

    eig_path = os.path.join(save_dir, f'eigenspectrum_{tag}.csv')
    with open(eig_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['re', 'im', 'k2']); w.writerows(rows)

    dom = np.array(dom)                                # (modes^2, 2): |k|, Re_dom
    # ---- power-law fit on k>0 shells with decaying dominant eigenvalue ----
    mask = (dom[:, 0] > 0) & (dom[:, 1] < 0)
    x = np.log(dom[mask, 0]); y = np.log(-dom[mask, 1])
    A = np.vstack([x, np.ones_like(x)]).T
    (p_hat, c), res, _, _ = np.linalg.lstsq(A, y, rcond=None)
    yfit = A @ np.array([p_hat, c])
    ss_res = float(((y - yfit) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2_pow = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # ---- quadratic (imposed-form) fit for comparison ----
    xq = dom[mask, 0] ** 2; yq = -dom[mask, 1]
    Aq = np.vstack([xq, np.ones_like(xq)]).T
    coef, _, _, _ = np.linalg.lstsq(Aq, yq, rcond=None)
    yqfit = Aq @ coef
    ss_res_q = float(((yq - yqfit) ** 2).sum())
    ss_tot_q = float(((yq - yq.mean()) ** 2).sum())
    r2_quad = 1.0 - ss_res_q / ss_tot_q if ss_tot_q > 0 else float('nan')

    p_learned = model.get_exponent()
    pct_stable = 100.0 * n_stable / n_total

    summ_path = os.path.join(save_dir, f'flexD_summary_{tag}.csv')
    with open(summ_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['Dparam', model.Dparam])
        w.writerow(['p_hat_loglog_fit', f'{p_hat:.4f}'])
        w.writerow(['r2_powerlaw', f'{r2_pow:.5f}'])
        w.writerow(['r2_quadratic', f'{r2_quad:.5f}'])
        w.writerow(['p_learned_param', f'{p_learned:.4f}'])
        w.writerow(['max_re_lambda', f'{max_re_global:.6f}'])
        w.writerow(['pct_stable', f'{pct_stable:.2f}'])
        w.writerow(['n_shells_fit', int(mask.sum())])

    print(f"\n{'='*62}\nFLEX-D ANALYSIS [{tag}]  (Dparam={model.Dparam})")
    print(f"  fitted exponent  p_hat = {p_hat:.3f}   (imposed form corresponds to 2)")
    print(f"  power-law R2 = {r2_pow:.4f} | quadratic-form R2 = {r2_quad:.4f}")
    if model.Dparam == 'freeexp':
        print(f"  learned exponent parameter p = {p_learned:.3f}")
    print(f"  max Re(lambda) = {max_re_global:.6f} | stable eigs = {pct_stable:.1f}%")
    print(f"  wrote {eig_path}\n  wrote {summ_path}\n{'='*62}")


# ============================================================
# Training (identical loop; logs learned exponent per epoch)
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

    model = SpatialKoopmanFlexD(
        modes=args.modes, width=args.width, r=args.r,
        T_in=args.T_in, H=H, W=W, Dparam=args.Dparam,
        fix_p=args.fix_p).to(device)

    ckpt_path = os.path.join(args.save_dir, f'best_koopman_{args.tag}.pt')

    if args.analyze:
        print(f"Loading {ckpt_path} for analysis ...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        analyze(model, args.tag, args.save_dir)
        return

    n_params = sum(p.numel() for p in model.parameters())
    print(f"FlexD ({args.Dparam}) params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, f'curve_koopman_{args.tag}.csv')
    with open(csv_path, 'w', newline='') as cf:
        csv.writer(cf).writerow(['epoch', 'train_l2', 'test_l2', 'max_re_eig',
                                 'stable', 'p', 'time_s'])

    best_test_l2 = float('inf')
    print(f"\n[{args.tag}] FLEX-D ({args.Dparam})  T_in={args.T_in} T_out={args.T_out} "
          f"r={args.r} modes={args.modes} epochs={args.epochs}\n")

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
            max_re = model.get_max_re_eigenvalue()
            stable = 1 if max_re <= 0 else 0
            p_now  = model.get_exponent()

        t1 = default_timer()

        with open(csv_path, 'a', newline='') as cf:
            csv.writer(cf).writerow([ep+1, f'{train_l2:.6f}', f'{test_l2:.6f}',
                                     f'{max_re:.6f}', stable, f'{p_now:.4f}',
                                     f'{t1-t0:.2f}'])

        if (ep + 1) % 10 == 0:
            flag = " <-- BEST" if test_l2 < best_test_l2 else ""
            pstr = f" p={p_now:.3f}" if args.Dparam == 'freeexp' else ""
            print(f"[{ep+1:04d}] train={train_l2:.4f} test={test_l2:.4f} "
                  f"Re(lam)_max={max_re:.5f}{pstr}{flag}")

        if test_l2 < best_test_l2:
            best_test_l2 = test_l2
            torch.save(model.state_dict(), ckpt_path)

    print(f"\n{'='*60}")
    print(f"DONE [FlexD {args.Dparam} {args.tag}]  best test_l2 = {best_test_l2:.4f}")
    if args.Dparam == 'freeexp':
        print(f"final learned exponent p = {model.get_exponent():.4f}  (imposed form = 2)")
    print(f"{'='*60}")
    print("Now run analysis:")
    print(f"  python navier_stokes/dissipation/lgn_flexD.py --data {args.data} --Dparam {args.Dparam} "
          f"--tag {args.tag} --save_dir {args.save_dir} --analyze")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     required=True)
    parser.add_argument('--Dparam',   choices=['freeexp', 'free', 'paper'], default='freeexp')
    parser.add_argument('--tag',      default='flexD_freeexp_nu1e5')
    parser.add_argument('--seed',     type=int,   default=0,
                        help='random seed (include it in --tag, e.g. _s1)')
    parser.add_argument('--fix_p',    type=float, default=None,
                        help='freeexp only: FREEZE the exponent at this value '
                             '(loss-vs-p control scan); everything else trains')
    parser.add_argument('--analyze',  action='store_true',
                        help='load best checkpoint and produce eigenspectrum + exponent fit')
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
    parser.add_argument('--save_dir', default='outputs/dissipation')
    args = parser.parse_args()
    set_seed(args.seed)
    train(args)
