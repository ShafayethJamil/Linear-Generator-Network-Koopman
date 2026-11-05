"""
Navier-Stokes ablations and sensitivity: Appendix A.2 (r-sweep, Fig. A.1), Appendix A.5 (structural
ablations, Table A.2, Fig. A.3) and Appendix A.10 (Table A.8), plus the ablation sentence of Section 3.1.

run(ck, check, check_pm, data_dir=None) is called by verify.py. Every expected value is the number as
printed in the paper. Checks recompute from the released checkpoints and training curves;
with data_dir they also re-evaluate test error on the Li et al. benchmark.

R^2 conventions, as in the scripts that produced each number:
  A.2 (navier_stokes/ablations/ablation_r.py, ablation_r2_fix.py): branch fits over the 143 wavevectors k != 0.
  Table A.2 "Dissipation R^2" (navier_stokes/extract_SD.py): dominant branch over all 144 wavevectors.
  Table A.8 (navier_stokes/ablations/analyze_spectral.py): all 144 wavevectors; SV-profile R^2 is the
  `compare` row of the paper model (r = 32, M = 12, raw input) on a 100-point normalised index grid.
"""
import csv, importlib.util, json, os, statistics as st
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.join(HERE, '..', '..', 'navier_stokes', 'ablations')


def _module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ABL, name + '.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _load(p):
    return torch.load(p, map_location='cpu', weights_only=False)


def _curve(p, col='test_l2'):
    return [float(row[col]) for row in csv.DictReader(open(p))]


def _generators(sd, kind='SD'):
    """L_k for every retained (k1, k2) >= 0 mode, rebuilt from the state dict as each model's get_generators()."""
    k2n = sd['k2_norm'].double().numpy(); M = k2n.shape[0]
    if kind == 'unconstrained':
        A = sd['A'].double().numpy(); b = sd['b'].double().numpy()
        Ls = [A + np.diag(b * k2n[i, j]) for i in range(M) for j in range(M)]
    elif kind == 'Sonly':
        P = sd['P'].double().numpy(); S = (P - P.T) / 2
        Ls = [S for i in range(M) for j in range(M)]
    else:
        d0 = F.softplus(sd['d'].double()).numpy(); a0 = F.softplus(sd['alpha'].double()).numpy()
        if kind == 'Donly':
            Ls = [-np.diag(d0 + a0 * k2n[i, j]) for i in range(M) for j in range(M)]
        else:
            P = sd['P'].double().numpy(); S = (P - P.T) / 2
            Ls = [S - np.diag(d0 + a0 * k2n[i, j]) for i in range(M) for j in range(M)]
    k2 = np.array([i * i + j * j for i in range(M) for j in range(M)], dtype=float)
    return np.array([np.linalg.eigvals(L) for L in Ls]), k2


def _r2(x, y):
    A = np.stack([np.ones_like(x), x], 1); c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(1 - ((y - A @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _branch_r2(eigs, k2, skip_dc):
    sel = k2 > 0 if skip_dc else np.ones_like(k2, bool)
    srt = np.sort(eigs.real, axis=1)[:, ::-1][sel]          # branch 0 = least damped (dominant)
    return np.array([_r2(k2[sel], srt[:, j]) for j in range(srt.shape[1])])


def _sv_profile(sd):
    P = sd['P'].double().numpy(); sv = np.linalg.svd((P - P.T) / 2, compute_uv=False)
    v = sv / sv[0]
    return np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(v)), v)


def run(ck, check, check_pm, data_dir=None):
    C = os.path.join(ck, 'ns_ablations')
    bool_ = lambda c: 1.0 if c else 0.0

    # ------------------------------------------------------------ A.2 r-sweep
    print("\nChannel-dimension sweep (Appendix A.2, Fig. A.1)")
    best = {(r, nu): min(_curve(f'{C}/r_sweep/r{r}_{nu}/curve.csv')) for r in [4, 8, 16, 32, 64] for nu in ['1e5', '1e3']}
    check('nu=1e-3 test L2, r=4 (training curve)', '0.201', best[(4, '1e3')], 'A.2')
    check('nu=1e-3 test L2, r=64 (training curve)', '0.042', best[(64, '1e3')], 'A.2')
    check('nu=1e-5 test L2, r=4 (training curve)', '0.302', best[(4, '1e5')], 'A.2')
    check('nu=1e-5 test L2, r=64 (training curve)', '0.232', best[(64, '1e5')], 'A.2')
    for nu in ['1e5', '1e3']:
        v = [best[(r, nu)] for r in [4, 8, 16, 32, 64]]
        check(f'accuracy improves monotonically with r, nu={nu} (1 = yes)', '1', bool_(all(a > b for a, b in zip(v, v[1:]))), 'A.2')
    dom = {}
    for r in [4, 8, 16, 32, 64]:
        for nu in ['1e5', '1e3']:
            e, k2 = _generators(_load(f'{C}/r_sweep/r{r}_{nu}/best.pt'))
            dom[(r, nu)] = _branch_r2(e, k2, skip_dc=True)
    check('dominant-branch R^2 > 0.88 for every r <= 32, both nu (1 = yes)', '1',
          bool_(min(dom[(r, nu)][0] for r in [4, 8, 16, 32] for nu in ['1e5', '1e3']) > 0.88), 'A.2')
    check('r=64, nu=1e-5: dominant-branch R^2', '0.753', dom[(64, '1e5')][0], 'A.2')
    check('r=64, nu=1e-5: median branch R^2', '0.990', float(np.median(dom[(64, '1e5')])), 'A.2')

    # ------------------------------------------------- A.5 structural ablations
    print("\nStructural ablations (Appendix A.5, Table A.2, Fig. A.3; Section 3.1)")
    S = f'{C}/structural'
    sd_l2 = [min(_curve(f'{S}/SD_s{s}_curve.csv')) for s in range(3)]
    check_pm('S-D test L2 over three seeds', '0.240', '0.0003', sd_l2, 'Table A.2')
    for kind, p_l2 in [('unconstrained', '0.240'), ('Sonly', '0.240'), ('Donly', '0.253')]:
        check(f'{kind}: test L2 (best epoch)', p_l2, min(_curve(f'{S}/{kind}_curve.csv')), 'Table A.2')
    E = {'SD': _generators(_load(f'{ck}/ns/lgnkm_nu1e5_s0.pt'))}
    for kind in ['unconstrained', 'Sonly', 'Donly']:
        E[kind] = _generators(_load(f'{S}/{kind}.pt'), kind)
    for kind, p_re, p_im, p_r2 in [('SD', '-0.021', '0.70', '0.92'), ('unconstrained', '0.347', '0.61', '0.94'),
                                   ('Sonly', '0', '0.62', None), ('Donly', '-0.026', '0', '0.80')]:
        e, k2 = E[kind]
        if kind == 'Donly':   # Table A.2 prints the final-epoch value here (best checkpoint: -0.0269)
            check('Donly: max Re(lambda), final epoch', p_re, _curve(f'{S}/Donly_curve.csv', 'max_re_eig')[-1], 'Table A.2')
        else:
            check(f'{kind}: max Re(lambda)', p_re, e.real.max(), 'Table A.2')
        check(f'{kind}: Im(lambda) range (+/-)', p_im, np.abs(e.imag).max(), 'Table A.2')
        if p_r2:
            check(f'{kind}: dissipation R^2 (dominant branch, all k)', p_r2, _r2(k2, e.real.max(axis=1)), 'Table A.2')
    check('stability row: S-D stable, unconstrained unstable, S-only marginal, D-only stable (1 = yes)', '1',
          bool_(E['SD'][0].real.max() < 0 and E['unconstrained'][0].real.max() > 0 and
                np.abs(E['Sonly'][0].real).max() < 1e-9 and E['Donly'][0].real.max() < 0), 'Table A.2')
    check('D-only: all eigenvalues real and negative (1 = yes)', '1',
          bool_(np.abs(E['Donly'][0].imag).max() == 0 and E['Donly'][0].real.max() < 0), 'A.5')
    unst = 100 * float((E['unconstrained'][0].real > 0).mean())
    check('unconstrained: % of eigenvalues unstable', '41', unst, 'Sec 3.1')
    check('unconstrained: % of eigenvalues unstable', '40', unst, 'Fig. A.3 caption')
    mr = _curve(f'{S}/unconstrained_curve.csv', 'max_re_eig')
    check('unconstrained: unstable at every training epoch (1 = yes)', '1', bool_(min(mr) > 0), 'A.5')
    tl = _curve(f'{S}/unconstrained_curve.csv', 'test_l2')
    i_best = int(np.argmin(tl))          # the reported checkpoint; training peaks higher (~0.39)
    check('unconstrained: max Re(lambda) at the reported checkpoint', '0.35', mr[i_best], 'A.5')

    # --------------------------------------------------- A.10 sensitivity
    print("\nSpectral readout vs r, M and input normalization (Appendix A.10, Table A.8)")
    cfg = [('r=4', f'{C}/r_sweep/r4_1e5/best.pt', '-0.005', '1.000', '0.999', '-1.60'),
           ('r=8', f'{C}/r_sweep/r8_1e5/best.pt', '-0.028', '0.993', '0.996', '0.45'),
           ('r=16', f'{C}/r_sweep/r16_1e5/best.pt', '-0.017', '0.997', '0.890', '0.958'),
           ('r=32 (paper)', f'{ck}/ns/lgnkm_nu1e5_s0.pt', '-0.021', '0.986', '0.920', '1'),
           ('r=64', f'{C}/r_sweep/r64_1e5/best.pt', '-0.016', '0.990', '0.756', '0.950'),
           ('M=8', f'{C}/sensitivity/M8.pt', '-0.016', '0.986', '0.960', '0.964'),
           ('M=16', f'{C}/sensitivity/M16.pt', '-0.035', '0.990', '0.484', '0.992'),
           ('normalized input', f'{C}/sensitivity/normalized_input.pt', '-0.006', '0.994', '0.826', '0.964')]
    ref = _sv_profile(_load(f'{ck}/ns/lgnkm_nu1e5_s0.pt'))
    stats = {}
    for name, path, p_re, p_med, p_dom, p_sv in cfg:
        sd = _load(path); e, k2 = _generators(sd); br = _branch_r2(e, k2, skip_dc=False)
        b = _sv_profile(sd); sv_r2 = float(1 - ((ref - b) ** 2).sum() / ((ref - ref.mean()) ** 2).sum())
        stats[name] = dict(max_re=e.real.max(), med=float(np.median(br)), frac=float((br > 0.9).sum()), n=len(br), sv=sv_r2)
        check(f'{name}: max Re(lambda)', p_re, e.real.max(), 'Table A.8')
        check(f'{name}: median branch R^2', p_med, stats[name]['med'], 'Table A.8')
        check(f'{name}: dominant branch R^2', p_dom, br[0], 'Table A.8')
        check(f'{name}: SV-profile R^2 vs paper model', p_sv, sv_r2, 'Table A.8')
    check('all configurations stable (1 = yes)', '1', bool_(max(s['max_re'] for s in stats.values()) < 0), 'A.10')
    meds = [s['med'] for s in stats.values()]
    check('median branch R^2, lowest over configurations', '0.986', min(meds), 'A.10')
    check('median branch R^2, highest over configurations', '1.000', max(meds), 'A.10')
    check('M=16: branches with R^2 > 0.9 (of 32)', '29', stats['M=16']['frac'], 'A.10')
    check('SV-profile R^2 for r >= 16, lowest ("~0.95 or above")', '0.95', min(stats[n]['sv'] for n in ['r=16', 'r=64']), 'A.10')

    # ------------------------------------------------- test error with data
    if data_dir:
        print("\nAblation test error on the Li et al. benchmark (Appendix A.2, Table A.2)")
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ab = _module('ablation_r')
        for nu, fname, T_out in [('1e5', 'NavierStokes_V1e-5_N1200_T20.mat', 10), ('1e3', 'ns_V1e-3_N5000_T50.mat', 40)]:
            _, _, x, y, grid, H, W = ab.load_data(os.path.join(data_dir, fname), 10, T_out, 1000, 200)
            for r, paper in ([(4, '0.302'), (64, '0.232')] if nu == '1e5' else [(4, '0.201'), (64, '0.042')]):
                m = ab.SpatialKoopmanLGN(12, 32, r, 10, H, W).to(dev)
                m.load_state_dict(_load(f'{C}/r_sweep/r{r}_{nu}/best.pt')); m.eval()
                check(f'nu={nu} r={r}: test L2 re-evaluated', paper, _eval(m, x, y, grid, T_out, ab.paper_rel_l2_loss, dev), 'A.2')
            if nu == '1e5':
                for kind, script, cls, paper in [('unconstrained', 'ablation_unconstrained_L', 'SpatialKoopmanUnconstrained', '0.240'),
                                                 ('Sonly', 'ablation_S_only', 'SpatialKoopmanSonly', '0.240'),
                                                 ('Donly', 'ablation_D_only', 'SpatialKoopmanDonly', '0.253')]:
                    mod = _module(script); m = getattr(mod, cls)(12, 32, 32, 10, H, W).to(dev)
                    m.load_state_dict(_load(f'{S}/{kind}.pt')); m.eval()
                    check(f'{kind}: test L2 re-evaluated', paper, _eval(m, x, y, grid, T_out, mod.paper_rel_l2_loss, dev), 'Table A.2')
            del x, y


def _eval(m, x, y, grid, T_out, loss, dev):
    tot = 0.0
    with torch.no_grad():
        for i in range(0, x.shape[0], 10):
            xb, yb = x[i:i + 10].to(dev), y[i:i + 10].to(dev).permute(0, 3, 1, 2)
            g = grid.to(dev).unsqueeze(0).expand(xb.shape[0], -1, -1, -1)
            tot += loss(m(xb, g, T_out), yb).item() * xb.shape[0]
    return tot / x.shape[0]
