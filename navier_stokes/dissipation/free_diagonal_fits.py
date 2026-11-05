"""
Appendix A.6, free diagonal D_k = diag(softplus(B_k)): does the learned dissipation
identify an exponent when no functional form in k is imposed?

For each free-diagonal checkpoint the channel-mean dissipation softplus(B).mean(-1)
is taken on the 12 x 12 retained wavevectors (integer |k|^2 = k1^2 + k2^2, 0..242).
  rho_shell   Spearman rank correlation of the |k|^2-shell average with |k|^2
  rho_wv      the same per wavevector (144 points)
  p(<=C)      exponent of an offset power law  y = c + a |k|^p  fitted to the
              shell averages with |k|^2 <= C (C = 20, 60, all); p_wv(<=C) per wavevector
The exponent moves with the window C, which is the A.6 finding.

Usage (from the repo root):
    python navier_stokes/dissipation/free_diagonal_fits.py --ckpt_dir checkpoints/dissipation
"""
import argparse, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

M = 12
_k1, _k2 = np.meshgrid(np.arange(M), np.arange(M), indexing='ij')
K2 = (_k1 ** 2 + _k2 ** 2).ravel()


def offset_fit(x, y, p_fixed=None):
    """Fit y = c + a*x^(p/2) with x = |k|^2 (so p is the |k| exponent).
    Returns (c, a, p, R^2). Offset REQUIRED (d0 floor).
    (Verbatim from the paper's analyze_spectral.py.)"""
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


def ckpt_path(ckpt_dir, nu, seed):
    return os.path.join(ckpt_dir, f"best_koopman_flexD_free_nu{nu}{'' if seed == 0 else f'_s{seed}'}.pt")


def stats(path):
    B = torch.load(path, map_location='cpu')['B']
    d = F.softplus(B.double()).mean(-1).numpy().ravel()          # channel-mean, per wavevector
    sh = pd.Series(d).groupby(K2).mean()
    x, y = sh.index.values.astype(float), sh.values
    out = dict(rho_shell=pd.Series(x).corr(pd.Series(y), method='spearman'),
               rho_wv=pd.Series(K2).corr(pd.Series(d), method='spearman'))
    for C, name in [(20, '20'), (60, '60'), (None, 'all')]:
        ms = x <= C if C else np.ones_like(x, bool)
        mw = K2 <= C if C else np.ones_like(K2, bool)
        out[f'p_shell_{name}'] = float(offset_fit(x[ms], y[ms])[2])
        out[f'p_wv_{name}'] = float(offset_fit(K2[mw].astype(float), d[mw])[2])
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', default='checkpoints/dissipation')
    a = ap.parse_args()
    for nu in ['1e5', '1e3']:
        for s in range(3):
            r = stats(ckpt_path(a.ckpt_dir, nu, s))
            print(f"nu={nu} seed {s}: rho shell {r['rho_shell']:.3f} wv {r['rho_wv']:.3f} | "
                  f"p shell <=20 {r['p_shell_20']:.2f}, <=60 {r['p_shell_60']:.2f}, all {r['p_shell_all']:.2f} | "
                  f"p wv <=20 {r['p_wv_20']:.2f}, <=60 {r['p_wv_60']:.2f}, all {r['p_wv_all']:.2f}")
