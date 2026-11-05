"""
Release: Table A.8 (Appendix A.10). Columns: max Re = max_re, median branch R^2 = r2_branch_median,
dominant branch R^2 = r2_dom_linear_fig3a, SV-profile R^2 = the `compare` matrix against sv_M12_r32 (row = paper model).
    C=checkpoints/ns_ablations
    python navier_stokes/ablations/analyze_spectral.py analyze --ckpt $C/r_sweep/r4_1e5/best.pt --r 4 --modes 12 --tag r4_M12 --save_dir runs/a10
    (likewise r8, r16, r64 from $C/r_sweep; r32: checkpoints/ns/lgnkm_nu1e5_s0.pt --tag M12_r32;
     M=8: $C/sensitivity/M8.pt --modes 8 --tag M8_r32; M=16: $C/sensitivity/M16.pt --modes 16 --tag M16_r32;
     normalized: $C/sensitivity/normalized_input.pt --modes 12 --tag M12_r32_norm)
    python navier_stokes/ablations/analyze_spectral.py compare --save_dir runs/a10 --profiles runs/a10/sv_M12_r32.csv runs/a10/sv_r4_M12.csv ...
The M = 8 and M = 16 models were trained by navier_stokes/train_lgnkm.py with --modes 8 / --modes 16 (tags nu1e5_M8, nu1e5_M16).

Original docstring:
E5 — SPECTRAL SENSITIVITY ANALYSIS (suVX Q4: spectral conclusions vs
latent dimension r, Fourier truncation M, normalization)
=====================================================================
Re-analysis of saved checkpoints — NO training. For each checkpoint it
computes the SPECTRAL metrics suVX asked about:

  - % stable eigenvalues, max Re(lambda)
  - dispersion readout: mean per-shell decay  ybar(|k|^2) = c + a*|k|^2
    fit R^2  (correct estimator per E1 notes: offset REQUIRED; and since
    tr S = 0, mean per-shell -Re(lambda) = mean diag D exactly)
  - offset power-law exponent p_hat (fit ybar = c + a*|k|^p)
  - Im-spectrum range and conjugate symmetry (skew-S sanity)
  - normalized SVD profile of S  -> written to CSV for cross-run comparison

Modes:
  analyze:  one row per checkpoint
    python analyze_spectral.py analyze --ckpt best_koopman_nu1e5_raw.pt \
        --r 32 --modes 12 --tag M12_r32
    python analyze_spectral.py analyze --ckpt best_koopman_nu1e5_M8.pt \
        --r 32 --modes 8 --tag M8_r32
    (repeat for every r-ablation / M-sweep / normalized checkpoint;
     rows append to E5_spectral_summary.csv)

  compare:  R^2 between normalized SV profiles of S across runs
    python analyze_spectral.py compare --profiles sv_M12_r32.csv sv_M8_r32.csv ...
    (profiles of different length are compared on a common normalized
     index grid, same convention as the paper's cross-viscosity Fig. 3b)

Outputs (--save_dir, default ./checkpoints):
  E5_spectral_summary.csv, sv_<tag>.csv, E5_sv_compare.csv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse, os, csv


# ============================================================
# Minimal generator reconstruction (encoder/decoder not needed)
# ============================================================

def load_generator(ckpt_path, r, modes, device='cpu'):
    """Rebuild S, D, and all L_k from a checkpoint's P/d/alpha entries."""
    sd = torch.load(ckpt_path, map_location=device)
    P     = sd['P']
    d     = sd['d']
    alpha = sd['alpha']
    assert P.shape == (r, r), f"P is {tuple(P.shape)}, expected ({r},{r}) — wrong --r?"

    S  = (P - P.T) / 2.0
    d0 = F.softplus(d)
    a0 = F.softplus(alpha)

    k1 = torch.arange(modes, dtype=torch.float32)
    kk1, kk2 = torch.meshgrid(k1, k1, indexing='ij')
    k2_phys = (kk1**2 + kk2**2)                              # physical |k|^2
    k2_norm = k2_phys / (modes**2)                           # model's buffer convention

    D_diag = d0.unsqueeze(0).unsqueeze(0) + \
             a0.unsqueeze(0).unsqueeze(0) * k2_norm.unsqueeze(-1)   # (M, M, r)
    L = S.unsqueeze(0).unsqueeze(0).expand(modes, modes, r, r).clone() \
        - torch.diag_embed(D_diag)
    return S, D_diag, L, k2_phys


def offset_fit(x, y, p_fixed=None):
    """Fit y = c + a*x^(p/2) with x = |k|^2 (so p is the |k| exponent).
    Returns (c, a, p, R^2). Offset REQUIRED (d0 floor)."""
    from scipy.optimize import curve_fit
    xk = np.sqrt(np.maximum(x, 0.0))
    if p_fixed is not None:
        f = lambda xk, c, a: c + a * np.power(xk, p_fixed)
        popt, _ = curve_fit(f, xk, y, p0=[max(y.min(), 1e-4), 1e-3], maxfev=50000)
        c, a, p = popt[0], popt[1], p_fixed
    else:
        f = lambda xk, c, a, p: c + a * np.power(xk, p)
        popt, _ = curve_fit(f, xk, y, p0=[max(y.min(), 1e-4), 1e-3, 2.0],
                            bounds=([0, 1e-10, 0.2], [np.inf, np.inf, 6.0]), maxfev=50000)
        c, a, p = popt
    yhat = c + a * np.power(xk, p)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return c, a, p, 1.0 - ss_res / (ss_tot + 1e-30)


def analyze(args):
    S, D_diag, L, k2_phys = load_generator(args.ckpt, args.r, args.modes)
    M, r = args.modes, args.r

    # ---- eigenvalues of every L_k ----
    eigs = torch.linalg.eigvals(L.reshape(-1, r, r))         # (M*M, r)
    re = eigs.real.numpy(); im = eigs.imag.numpy()
    n_tot = re.size
    pct_stable = 100.0 * float((re <= 1e-9).sum()) / n_tot
    max_re = float(re.max())
    # conjugate symmetry: for a symmetric multiset, sort(im) == sort(-im),
    # so their DIFFERENCE is ~0 (a sum would report ~2*max|Im| even when perfect)
    im_sym = float(np.abs(np.sort(im.ravel()) - np.sort(-im.ravel())).max())

    k2_flat = k2_phys.reshape(-1).numpy()

    # ---- dispersion readout A (Fig-3a estimator, paper-comparable) ----
    # dominant eigenvalue = slowest decay = max Re per wavevector;
    # LINEAR fit Re(lam_dom) vs |k|^2 with intercept, per-wavevector points,
    # exactly the quantity behind the paper's R^2 = 0.926 / 0.980.
    dom_re = re.max(axis=1)                                  # (M*M,) max Re per wavevector
    A = np.stack([np.ones_like(k2_flat), k2_flat], axis=1)
    coef, _, _, _ = np.linalg.lstsq(A, dom_re, rcond=None)
    yhat = A @ coef
    ss_res = float(((dom_re - yhat) ** 2).sum())
    ss_tot = float(((dom_re - dom_re.mean()) ** 2).sum())
    r2_dom_linear = 1.0 - ss_res / (ss_tot + 1e-30)
    dom_slope = float(coef[1])

    # ---- dispersion readout B (branch-resolved: THE robust statistic) ----
    # sort eigenvalues by Re at each wavevector -> branch j = j-th slowest;
    # linear fit per branch; median R^2 over all r branches is robust to a
    # single slow channel capturing the dominant envelope (seen at M=16).
    eig_sorted = np.sort(re, axis=1)[:, ::-1]                # (M*M, r), branch 0 dominant
    branch_r2 = []
    for j in range(r):
        yb = eig_sorted[:, j]
        cb, _, _, _ = np.linalg.lstsq(A, yb, rcond=None)
        yhb = A @ cb
        ssr = float(((yb - yhb) ** 2).sum())
        sst = float(((yb - yb.mean()) ** 2).sum())
        branch_r2.append(1.0 - ssr / (sst + 1e-30))
    branch_r2 = np.array(branch_r2)
    r2_branch_median = float(np.median(branch_r2))
    frac_branch_09   = float((branch_r2 > 0.9).mean())

    # ---- dispersion readout C (E1 mean-shell estimator; TAUTOLOGICAL for
    #      imposed-form checkpoints — do not quote for those) ----
    mean_decay = (-re).mean(axis=1)                          # = mean diag D exactly (tr S = 0)
    shells = {}
    for k2v, y in zip(k2_flat, mean_decay):
        shells.setdefault(round(float(k2v), 6), []).append(y)
    k2_sh = np.array(sorted(shells.keys()))
    y_sh  = np.array([np.mean(shells[k]) for k in k2_sh])

    _, _, _, r2_quad = offset_fit(k2_sh, y_sh, p_fixed=2.0)
    _, _, p_hat, r2_pow = offset_fit(k2_sh, y_sh)

    # ---- SVD profile of S (normalized) ----
    sv = torch.linalg.svdvals(S).numpy()
    sv_n = sv / (sv[0] + 1e-30)
    pair_gap = float(np.abs(sv[0::2][:len(sv[1::2])] - sv[1::2]).max())  # skew => paired SVs

    os.makedirs(args.save_dir, exist_ok=True)
    sv_path = os.path.join(args.save_dir, f'sv_{args.tag}.csv')
    with open(sv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['index', 'sv', 'sv_normalized'])
        for i, (v, vn) in enumerate(zip(sv, sv_n)):
            w.writerow([i, f'{v:.8f}', f'{vn:.8f}'])

    summ_path = os.path.join(args.save_dir, 'E5_spectral_summary.csv')
    new = not os.path.exists(summ_path)
    with open(summ_path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['tag', 'ckpt', 'r', 'modes', 'n_eigs', 'pct_stable', 'max_re',
                        'r2_branch_median', 'frac_branch_r2_gt0.9',
                        'r2_dom_linear_fig3a', 'dom_slope',
                        'r2_quadratic_meanshell', 'p_hat', 'r2_powerlaw', 'im_range',
                        'im_conj_sym_err', 'sv_pair_gap', 'norm_S_frobenius'])
        w.writerow([args.tag, os.path.basename(args.ckpt), r, M, n_tot,
                    f'{pct_stable:.2f}', f'{max_re:.6f}',
                    f'{r2_branch_median:.4f}', f'{frac_branch_09:.3f}',
                    f'{r2_dom_linear:.4f}', f'{dom_slope:.3e}',
                    f'{r2_quad:.4f}',
                    f'{p_hat:.3f}', f'{r2_pow:.4f}', f'{float(np.abs(im).max()):.4f}',
                    f'{im_sym:.2e}', f'{pair_gap:.2e}',
                    f'{float(torch.linalg.norm(S)):.4f}'])

    print(f"[{args.tag}] r={r} M={M}  n_eigs={n_tot}")
    print(f"  stability : {pct_stable:.2f}% stable, max Re = {max_re:.5f}")
    print(f"  BRANCHES  : median R^2 = {r2_branch_median:.4f}, "
          f"{100*frac_branch_09:.0f}% of branches > 0.9   <- headline statistic")
    print(f"  Fig-3a    : dominant-branch linear R^2 = {r2_dom_linear:.4f} "
          f"(slope {dom_slope:.2e})   <- envelope; fragile at large r/M")
    print(f"  mean-shell: quadratic R^2 = {r2_quad:.4f}, p_hat = {p_hat:.2f} "
          f"(TAUTOLOGICAL for imposed-form ckpts — ignore for those)")
    print(f"  Im range  : +/-{np.abs(im).max():.3f} (conjugate-symmetry err {im_sym:.1e})")
    print(f"  S         : ||S||_F = {torch.linalg.norm(S):.3f}, SV pair gap {pair_gap:.1e}")
    print(f"  appended to {summ_path}; SV profile -> {sv_path}")


def compare(args):
    profs = {}
    for path in args.profiles:
        tag = os.path.basename(path).replace('sv_', '').replace('.csv', '')
        rows = list(csv.DictReader(open(path)))
        profs[tag] = np.array([float(r['sv_normalized']) for r in rows])

    grid = np.linspace(0, 1, 100)
    interp = {t: np.interp(grid, np.linspace(0, 1, len(v)), v) for t, v in profs.items()}

    tags = list(interp.keys())
    out = os.path.join(args.save_dir, 'E5_sv_compare.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([''] + tags)
        print("\nSV-profile R^2 matrix (normalized index grid, as in Fig. 3b):")
        print('            ' + '  '.join(f'{t:>12s}' for t in tags))
        for ti in tags:
            row = [ti]
            line = f'{ti:>12s}'
            for tj in tags:
                a, b = interp[ti], interp[tj]
                ss_res = float(((a - b) ** 2).sum())
                ss_tot = float(((a - a.mean()) ** 2).sum())
                r2 = 1.0 - ss_res / (ss_tot + 1e-30)
                row.append(f'{r2:.4f}')
                line += f'  {r2:12.4f}'
            w.writerow(row)
            print(line)
    print(f"wrote {out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    pa = sub.add_parser('analyze')
    pa.add_argument('--ckpt',  required=True)
    pa.add_argument('--r',     type=int, required=True)
    pa.add_argument('--modes', type=int, required=True)
    pa.add_argument('--tag',   required=True)
    pa.add_argument('--save_dir', default='./checkpoints')

    pc = sub.add_parser('compare')
    pc.add_argument('--profiles', nargs='+', required=True)
    pc.add_argument('--save_dir', default='./checkpoints')

    args = p.parse_args()
    if args.cmd == 'analyze':
        analyze(args)
    else:
        compare(args)
