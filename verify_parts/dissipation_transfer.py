"""
verify.py part: dissipation form (Section 3.2, Table A.3, Appendix A.6) and
cross-viscosity transfer (Section 4.2, Fig. 6, Table A.1).

run(ck, check, check_pm, data_dir=None)
  ck        checkpoints/ folder (uses dissipation/, transfer/, ns/lgnkm_nu1e5_s0.pt)
  data_dir  optional folder with the Li et al. .mat files; when given, the test
            errors are also re-evaluated from the checkpoints (GPU recommended)

Learned p (Table A.3) is the final-epoch value from the training curves; test
errors are the minimum over epochs (the saved checkpoint). Transfer is seed 0.
"""
import csv, os, re, statistics as st, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DRES = os.path.join(ROOT, 'navier_stokes', 'dissipation', 'results')
TRES = os.path.join(ROOT, 'transfer', 'results')
SUF = ['', '_s1', '_s2']


def _curve(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _flex(v, nu, s):
    return _curve(os.path.join(DRES, f'curve_koopman_flexD_{v}_nu{nu}{SUF[s]}.csv'))


def run(ck, check, check_pm, data_dir=None):
    sys.path.insert(0, os.path.join(ROOT, 'navier_stokes', 'dissipation'))
    sys.path.insert(0, os.path.join(ROOT, 'transfer'))
    import free_diagonal_fits as fdf
    import fig6_metrics

    print("\nDissipation form: learnable exponent and free diagonal (Section 3.2, Table A.3, A.6)")
    p_fin = {nu: [float(_flex('freeexp', nu, s)[-1]['p']) for s in range(3)] for nu in ['1e5', '1e3']}
    l2_exp = {nu: [min(float(r['test_l2']) for r in _flex('freeexp', nu, s)) for s in range(3)] for nu in ['1e5', '1e3']}
    l2_free = {nu: [min(float(r['test_l2']) for r in _flex('free', nu, s)) for s in range(3)] for nu in ['1e5', '1e3']}
    l2_fix = {nu: min(float(r['test_l2']) for r in _curve(os.path.join(DRES, f'curve_fixed_p2_nu{nu}_s0.csv')))
              for nu in ['1e5', '1e3']}
    check_pm('learned p, nu=1e-5, final epoch, 3 seeds', '2.39', '0.02', p_fin['1e5'], 'Sec 3.2')
    check_pm('learned p, nu=1e-3, final epoch, 3 seeds', '2.16', '0.12', p_fin['1e3'], 'Sec 3.2')
    check('learned-p test L2 mean, nu=1e-5', '0.241', st.mean(l2_exp['1e5']), 'Sec 3.2')
    check('fixed p=2 test L2, nu=1e-5 (seed 0)', '0.240', l2_fix['1e5'], 'Sec 3.2')
    check('learned-p test L2 mean, nu=1e-3', '0.062', st.mean(l2_exp['1e3']), 'Sec 3.2')
    check('fixed p=2 test L2, nu=1e-3 (seed 0)', '0.063', l2_fix['1e3'], 'Sec 3.2')
    for nu, pp, ll in [('1e5', ['2.377', '2.388', '2.419'], ['0.2407', '0.2408', '0.2406']),
                       ('1e3', ['2.281', '2.033', '2.166'], ['0.0615', '0.0617', '0.0617'])]:
        for s in range(3):
            check(f'learned p, nu={nu}, seed {s}', pp[s], p_fin[nu][s], 'Table A.3')
            check(f'learned-p test L2, nu={nu}, seed {s}', ll[s], l2_exp[nu][s], 'Table A.3')
    check('fixed p=2 test L2, nu=1e-5', '0.2397', l2_fix['1e5'], 'Table A.3')
    check('fixed p=2 test L2, nu=1e-3', '0.0627', l2_fix['1e3'], 'Table A.3')
    for s in (1, 2):   # superseded runs launched with T_out = 10 (A.6, horizon dependence)
        log = open(os.path.join(DRES, f'log_flexD_freeexp_nu1e3_Tout10_s{s}.log')).read()
        assert 'T_out=10' in log
        p10 = float(re.search(r'final learned exponent p = ([0-9.]+)', log).group(1))
        check(f'learned p with T_out=10, nu=1e-3, seed {s}', '1.2', p10, 'A.6')
    for nu, ll in [('1e5', ['0.2369', '0.2385', '0.2397']), ('1e3', ['0.0598', '0.0523', '0.0589'])]:
        for s in range(3):
            check(f'free-diagonal test L2, nu={nu}, seed {s}', ll[s], l2_free[nu][s], 'A.6')

    fs = {nu: [fdf.stats(fdf.ckpt_path(os.path.join(ck, 'dissipation'), nu, s)) for s in range(3)]
          for nu in ['1e5', '1e3']}
    check('free D: Spearman rho, shell-averaged, nu=1e-5 (3-seed mean)', '0.95',
          st.mean(r['rho_shell'] for r in fs['1e5']), 'A.6')
    check('free D: Spearman rho per wavevector, nu=1e-3, min', '0.70', min(r['rho_wv'] for r in fs['1e3']), 'A.6')
    check('free D: Spearman rho per wavevector, nu=1e-3, max', '0.73', max(r['rho_wv'] for r in fs['1e3']), 'A.6')
    for key, lo, hi, lab in [('p_shell_20', '3.3', '5.2', '|k|^2 <= 20'), ('p_shell_60', '1.8', '2.6', '|k|^2 <= 60'),
                             ('p_shell_all', '1.0', '1.4', 'all modes')]:
        v = [r[key] for r in fs['1e5']]
        check(f'free D exponent, nu=1e-5, {lab}, seed min', lo, min(v), 'A.6')
        check(f'free D exponent, nu=1e-5, {lab}, seed max', hi, max(v), 'A.6')
    v = [r['p_shell_60'] for r in fs['1e3']]   # same estimator as the nu=1e-5 ranges
    check('free D exponent, nu=1e-3, |k|^2 <= 60 (shell fit), seed min', '1.0', min(v), 'A.6')
    check('free D exponent, nu=1e-3, |k|^2 <= 60 (shell fit), seed max', '1.6', max(v), 'A.6')

    print("\nCross-viscosity transfer, nu=1e-5 -> 1e-3, seed 0 (Section 4.2, Fig. 6, Table A.1)")
    m = {r['N']: r for r in fig6_metrics.metrics(TRES)}
    check('N=50: epoch freeze-S first reaches the scratch best', '71', m[50]['freezeS_reaches_scratch_best'], 'Sec 4.2')
    check('N=50: convergence speedup (~6x)', '6', m[50]['speedup'], 'Sec 4.2')
    check('N=50: lower final error than scratch (%)', '18', m[50]['gain_pct'], 'Sec 4.2')
    check('freeze-S test L2 at N=50', '0.191', m[50]['freezeS'], 'Sec 4.2')
    check('scratch test L2 at N=100', '0.183', m[100]['scratch'], 'Sec 4.2')
    check('N <= 500 budgets where transfer beats scratch (of 4)', '4',
          sum(m[n]['gain_pct'] > 0 for n in (50, 100, 200, 500)), 'Sec 4.2')
    for n, f, a, d in zip(fig6_metrics.BUDGETS, ['0.191', '0.152', '0.121', '0.087', '0.069'],
                          ['0.182', '0.147', '0.118', '0.085', '0.066'], ['4.5', '4.0', '2.5', '2.0', '4.0']):
        check(f'Table A.1 N={n}: freeze-S', f, m[n]['freezeS'], 'Table A.1')
        check(f'Table A.1 N={n}: transfer-all', a, m[n]['transfer_all'], 'Table A.1')
        check(f'Table A.1 N={n}: Delta (%)', d, m[n]['delta_pct'], 'Table A.1')
    check('budgets with freeze-S within 5% of transfer-all (of 5)', '5',
          sum(m[n]['delta_pct'] < 5 for n in fig6_metrics.BUDGETS), 'A.3')
    for n, sp, g in zip(fig6_metrics.BUDGETS, ['6.03', '4.66', '2.99', '1.55', None],
                        ['18.1', '16.8', '12.3', '3.3', '-11.4']):
        if sp:
            check(f'Fig 6(c) N={n}: speedup', sp, m[n]['speedup'], 'Fig 6(c), plotted')
        check(f'Fig 6(c) N={n}: gain over scratch (%)', g, m[n]['gain_pct'], 'Fig 6(c), plotted')
    P0 = torch.load(os.path.join(ck, 'ns', 'lgnkm_nu1e5_s0.pt'), map_location='cpu')['P']
    same = sum(torch.equal(torch.load(os.path.join(ck, 'transfer', f'best_transfer_freezeS_N{n}_s0.pt'),
                                      map_location='cpu')['P'], P0) for n in fig6_metrics.BUDGETS)
    check('freeze-S checkpoints whose S equals the pretrained S exactly (of 5)', '5', same, 'Sec 4.2')

    if data_dir:
        _reeval(ck, check, data_dir, l2_exp, l2_free, m)


def _eval(model, x, y, grid, T_out, dev):
    sys.path.insert(0, os.path.join(ROOT, 'navier_stokes', 'dissipation'))
    from lgn_flexD import paper_rel_l2_loss
    err = 0.0
    with torch.no_grad():
        for i in range(0, x.shape[0], 10):
            xb, yb = x[i:i + 10].to(dev), y[i:i + 10].to(dev).permute(0, 3, 1, 2)
            g = grid.to(dev).unsqueeze(0).expand(xb.shape[0], -1, -1, -1)
            err += paper_rel_l2_loss(model(xb, g, T_out), yb).item() * xb.shape[0]
    return err / x.shape[0]


def _reeval(ck, check, data_dir, l2_exp, l2_free, m):
    """Re-evaluate the saved checkpoints on the benchmark test split (trajectories 1000:1200).
    Passes when the re-evaluated error matches the logged curve minimum to < 5e-5
    (GPU nondeterminism is ~1e-5)."""
    import lgn_flexD, transfer_experiment
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nRe-evaluated from checkpoints on the test split (dissipation and transfer)")
    files = {'1e5': ('NavierStokes_V1e-5_N1200_T20.mat', 10), '1e3': ('ns_V1e-3_N5000_T50.mat', 40)}
    for nu, (fname, T_out) in files.items():
        _, _, x, y, grid, H, W = lgn_flexD.load_data(os.path.join(data_dir, fname), 10, T_out, 1000, 200)
        for v, ref in [('freeexp', l2_exp), ('free', l2_free)]:
            for s in range(3):
                mdl = lgn_flexD.SpatialKoopmanFlexD(12, 32, 32, 10, H, W, Dparam=v).to(dev)
                mdl.load_state_dict(torch.load(os.path.join(ck, 'dissipation',
                                    f'best_koopman_flexD_{v}_nu{nu}{SUF[s]}.pt'), map_location=dev)); mdl.eval()
                check(f'{v} nu={nu} seed {s}: |ckpt - curve min| x 1e4', '0',
                      1e4 * abs(_eval(mdl, x, y, grid, T_out, dev) - ref[nu][s]), 'Table A.3 / A.6')
        if nu == '1e3':
            for n in fig6_metrics_budgets():
                for c, key in [('scratch', 'scratch'), ('transfer_freezeS', 'freezeS'), ('transfer_all', 'transfer_all')]:
                    mdl = transfer_experiment.SpatialKoopmanLGN(12, 32, 32, 10, H, W).to(dev)
                    mdl.load_state_dict(torch.load(os.path.join(ck, 'transfer', f'best_{c}_N{n}_s0.pt'),
                                                   map_location=dev)); mdl.eval()
                    check(f'transfer {c} N={n}: |ckpt - curve min| x 1e4', '0',
                          1e4 * abs(_eval(mdl, x, y, grid, T_out, dev) - m[n][key]), 'Table A.1 / Fig 6(b)')


def fig6_metrics_budgets():
    import fig6_metrics
    return fig6_metrics.BUDGETS
