"""
Reproduce the paper's headline numbers from the released checkpoints.

    python verify.py                       # checkpoint-only checks, CPU, < 1 min
    python verify.py --data_dir data/ns    # also re-evaluates test error on the
                                           # Li et al. benchmark (GPU recommended)

Each check prints the value as printed in the paper, the recomputed value, and PASS
when the recomputed value rounds to the printed one. Exit code 1 if any check fails.
"""
import argparse, json, os, sys, statistics as st
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
results = []


def check(name, paper, value, where):
    """paper: the number as printed (string); PASS if value rounds to it."""
    dec = len(paper.split('.')[1]) if '.' in paper else 0
    ok = round(float(value), dec) == float(paper)
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:58s} paper {paper:>8s}   recomputed {value:.{dec + 2}f}   [{where}]")


def check_pm(name, paper_mean, paper_sd, values, where):
    check(name + ' (mean)', paper_mean, st.mean(values), where)
    check(name + ' (s.d.)', paper_sd, st.stdev(values), where)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def S_of(sd):
    P = sd['P'].double().numpy()
    return (P - P.T) / 2


# ---------------------------------------------------------------- Navier-Stokes
def ns_eigs(sd):
    """All eigenvalues of L_k = S - D_k over the retained M x M modes, and |k|^2 per mode."""
    S = S_of(sd)
    d0 = F.softplus(sd['d'].double()).numpy()
    a0 = F.softplus(sd['alpha'].double()).numpy()
    k2n = sd['k2_norm'].double().numpy()
    M = k2n.shape[0]
    eigs, k2 = [], []
    for i in range(M):
        for j in range(M):
            eigs.append(np.linalg.eigvals(S - np.diag(d0 + a0 * k2n[i, j])))
            k2.append(i * i + j * j)
    return np.array(eigs), np.array(k2)


def r2_identity(x, y):
    """Paper Sec. 3.2 convention: R^2 of [0,1]-normalised profile x against reference y."""
    xn = (x - x.min()) / (x.max() - x.min()); yn = (y - y.min()) / (y.max() - y.min())
    return float(1 - ((xn - yn) ** 2).sum() / ((yn - yn.mean()) ** 2).sum())


def verify_ns(ck):
    print("\nNavier-Stokes generator (Sections 3.1-3.2, Appendix A.10)")
    E = {nu: ns_eigs(load(f'{ck}/ns/lgnkm_nu{nu}_s0.pt')) for nu in ['1e5', '1e3']}
    e5, k5 = E['1e5']; e3, k3 = E['1e3']
    check('eigenvalues per model (r x M^2)', '4608', e5.size, 'seed 0')
    check('max Re(lambda), both models', '-0.0211', max(e5.real.max(), e3.real.max()), 'seed 0')
    check('nu=1e-3: min Re(lambda)', '-0.38', e3.real.min(), 'Sec 3.1')
    check('nu=1e-3: max Re(lambda)', '-0.04', e3.real.max(), 'Sec 3.1')
    check('nu=1e-3: max |Im(lambda)|', '0.32', np.abs(e3.imag).max(), 'Sec 3.1')
    check('nu=1e-5: min Re(lambda)', '-1.76', e5.real.min(), 'Sec 3.1')
    check('nu=1e-5: max |Im(lambda)|', '0.70', np.abs(e5.imag).max(), 'Sec 3.1')
    check('decay-rate range ratio, 1e-5 / 1e-3', '5.2', np.ptp(e5.real) / np.ptp(e3.real), 'Sec 3.1')
    check('oscillation range ratio, 1e-5 / 1e-3', '2.2', np.ptp(e5.imag) / np.ptp(e3.imag), 'Sec 3.1')
    dom = e5.real.max(axis=1)                     # least-damped eigenvalue at each wavevector
    check('dominant-branch R^2, Re(lambda) vs |k|^2, nu=1e-5', '0.920',
          np.corrcoef(k5, dom)[0, 1] ** 2, 'A.10 r=32')

    # cross-viscosity, three same-seed pairs (nu=1e-5 seed s vs nu=1e-3 seed s)
    sd = {(nu, s): load(f'{ck}/ns/lgnkm_nu{nu}_s{s}.pt') for nu in ['1e5', '1e3'] for s in range(3)}
    cos, sv, dd, aa, fro = [], [], [], [], []
    for s in range(3):
        A, B = sd[('1e5', s)], sd[('1e3', s)]
        SA, SB = S_of(A), S_of(B)
        cos.append(float(SA.ravel() @ SB.ravel() / np.linalg.norm(SA) / np.linalg.norm(SB)))
        sv.append(r2_identity(np.linalg.svd(SB, compute_uv=False), np.linalg.svd(SA, compute_uv=False)))
        sort = lambda t: np.sort(F.softplus(t.double()).numpy())
        dd.append(r2_identity(sort(B['d']), sort(A['d'])))
        aa.append(r2_identity(sort(B['alpha']), sort(A['alpha'])))
        fro.append(np.linalg.norm(SA) / np.linalg.norm(SB))
    check_pm('raw S cosine similarity across viscosities', '0.067', '0.058', cos, 'Sec 3.2, 3 seed pairs')
    check_pm('S singular-value profile R^2 across viscosities', '0.870', '0.063', sv, 'Sec 3.2, 3 seed pairs')
    check_pm('sorted base damping d R^2', '0.960', '0.017', dd, 'Sec 3.2, 3 seed pairs')
    check_pm('sorted Laplacian coefficient alpha R^2', '0.585', '0.273', aa, 'Sec 3.2, 3 seed pairs')
    check_pm('||S||_F ratio, 1e-5 / 1e-3', '1.82', '0.14', fro, 'Sec 3.1, 3 seed pairs')


# ------------------------------------------------------------------------- NLSE
OMEGA15 = 2 * np.pi * 15 / 100.0      # rad/ps: band edge of 16 rfft modes over a 100 ps window
DZ = 0.166                            # km per snapshot


def verify_nlse(ck):
    print("\nNonlinear Schroedinger equation (Section 3.3, Appendix A.8)")
    const = 2 / (OMEGA15 ** 2 * DZ)
    check('derived readout constant 2 / (Omega_15^2 dz)', '13.56', const, 'Sec 3.3')
    paper = {10: ['9.67', '10.24', '9.20'], 15: ['14.67', '14.20', '15.18'], 20: ['19.95', '18.76', '19.42'],
             25: ['25.41', '25.05', '24.39'], 30: ['29.36', '29.97', '28.86']}
    ratios, errs = [], []
    for b, pv in paper.items():
        est = []
        for s in range(3):
            fmax = np.abs(np.linalg.eigvals(S_of(load(f'{ck}/nlse/b2_{b}_s{s}.pt'))).imag).max()
            est.append(const * fmax); ratios.append(est[-1] / b)
            check(f'|beta2| = {b}, seed {s}: 13.56 f_max', pv[s], est[-1], 'A.8 table')
        errs.append(abs(st.mean(est) - b) / b)
    check('largest error of the three-seed mean (%)', '3.1', 100 * max(errs), 'Sec 3.3')
    check('f_max / predicted band-edge advance, minimum', '0.92', min(ratios), 'A.8, 15 models')
    check('f_max / predicted band-edge advance, maximum', '1.02', max(ratios), 'A.8, 15 models')
    f5 = np.abs(np.linalg.eigvals(S_of(load(f'{ck}/nlse/b2_5_s0.pt'))).imag).max()
    check('|beta2| = 5 (below the lower bound): reads', '6.5', const * f5, 'A.8 (i)')


# ------------------------------------------------------------- test error (data)
def verify_accuracy(ck, data_dir):
    print("\nTest relative L2 on the Li et al. benchmark (Table 1)")
    sys.path.insert(0, os.path.join(HERE, 'navier_stokes'))
    import train_lgnkm, train_fno
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for nu, fname, T_out, p_lgn, p_fno in [('1e5', 'NavierStokes_V1e-5_N1200_T20.mat', 10, '0.240', '0.158'),
                                            ('1e3', 'ns_V1e-3_N5000_T50.mat', 40, '0.063', '0.019')]:
        _, _, x, y, grid, H, W = train_lgnkm.load_data(os.path.join(data_dir, fname), 10, T_out, 1000, 200)
        lgn = train_lgnkm.SpatialKoopmanLGN(12, 32, 32, 10, H, W).to(dev)
        lgn.load_state_dict(load(f'{ck}/ns/lgnkm_nu{nu}_s0.pt')); lgn.eval()
        fno = train_fno.FNO2d(12, 12, 20, 10).to(dev)
        fno.load_state_dict(load(f'{ck}/ns/fno_nu{nu}.pt')); fno.eval()
        err = {'lgn': 0.0, 'fno': 0.0}
        with torch.no_grad():
            for i in range(0, x.shape[0], 10):
                xb, yb = x[i:i + 10].to(dev), y[i:i + 10].to(dev).permute(0, 3, 1, 2)
                g = grid.to(dev).unsqueeze(0).expand(xb.shape[0], -1, -1, -1)
                err['lgn'] += train_lgnkm.paper_rel_l2_loss(lgn(xb, g, T_out), yb).item() * xb.shape[0]
                err['fno'] += train_lgnkm.paper_rel_l2_loss(fno.rollout(xb, g, T_out), yb).item() * xb.shape[0]
        check(f'LGN-KM test rel. L2, nu={nu} ({T_out} steps)', p_lgn, err['lgn'] / x.shape[0], 'Table 1')
        check(f'FNO test rel. L2, nu={nu} ({T_out} steps)', p_fno, err['fno'] / x.shape[0], 'Table 1')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', default=os.path.join(HERE, 'checkpoints'))
    ap.add_argument('--data_dir', default=None, help='folder with the two Li et al. .mat files')
    a = ap.parse_args()
    if not os.path.isfile(os.path.join(a.ckpt_dir, 'MANIFEST.json')):
        sys.exit(f"No checkpoints in {a.ckpt_dir}; run scripts/download_checkpoints.sh first.")
    verify_ns(a.ckpt_dir)
    verify_nlse(a.ckpt_dir)
    if a.data_dir:
        verify_accuracy(a.ckpt_dir, a.data_dir)
    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail} of {len(results)} checks pass")
    sys.exit(1 if n_fail else 0)
