"""
verify part: baselines, long horizon, and continuous time
(Table 1, Section 4.1 + Fig. 5, Appendix A.9 Tables A.6 and A.7).

run(ck, check, check_pm, data_dir=None) is called by verify.py. Values are compared
with the numbers as printed in the paper. Sources, by check:
  - LRAN spectral radius: recomputed from the released checkpoints.
  - DMD / LRAN / KNO test errors, DMD stability: the shipped result CSVs written by
    baselines/*.py (DMD has no checkpoint; it is deterministic given the data).
  - Tables A.6 / A.7, blow-up times, enstrophy: the shipped CSVs of the Nov 3 2026 run
    (long_horizon/results/), and its energy-gate JSONs.
  - Fig. 5c timing: the shipped timing CSV (wall-clock numbers cannot be recomputed
    portably).
With data_dir (the Li et al. benchmark folder), Fig. 5a/5b are recomputed from the
checkpoints. If the regenerated long-horizon files are present (data_dir,
outputs/long_horizon/, or $LGNKM_LONG_DIR), Tables A.6/A.7 are re-evaluated too.
"""
import csv, glob, json, math, os, statistics as st, subprocess, sys, tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BR = os.path.join(HERE, 'baselines', 'results')
LH = os.path.join(HERE, 'long_horizon', 'results')


def _kv(path):
    return {r['metric']: float(r['value']) for r in csv.DictReader(open(path))}


def _rows(path):
    return list(csv.DictReader(open(path)))


def _sci(check, name, paper_mantissa, exp, value, where):
    check(f'{name} (in units of 1e{exp})', paper_mantissa, value / 10 ** exp, where)


def _table_A6(check, d, label):
    """Table A.6 and the Sec. 4.1 statements that come from the same run."""
    r = {int(x['t']): x for x in _rows(os.path.join(d, 'E3_per_step.csv'))}
    paper = {'lgn': ['0.16', '0.36', '1.06', '1.11', '1.04', '1.01', '1.00'],
             'kno': ['0.12', '0.24', '0.84', '1.19', '1.20', '1.11', '1.03'],
             'fno': ['0.11', '0.23', '0.86', '1.51', '1.59', '72.7', None],
             'lran': ['0.17', '0.37', '14.0', '249', None, None, 'inf']}
    for model, vals in paper.items():
        for t, pv in zip([15, 20, 30, 40, 50, 100, 200], vals):
            if pv is not None:
                check(f'Table A.6 {model.upper()} t={t}', pv, float(r[t][f'{model}_rel_l2']), label)
    _sci(check, 'Table A.6 FNO t=200', '1.6', 7, float(r[200]['fno_rel_l2']), label)
    _sci(check, 'Table A.6 LRAN t=50', '7.6', 3, float(r[50]['lran_rel_l2']), label)
    _sci(check, 'Table A.6 LRAN t=100', '6.4', 10, float(r[100]['lran_rel_l2']), label)
    T = sorted(r)
    lg = [float(r[t]['lgn_rel_l2']) for t in T]; fn = [float(r[t]['fno_rel_l2']) for t in T]
    first = next(t for t, a, b in zip(T, lg, fn) if a < b)
    stays = all(a < b for t, a, b in zip(T, lg, fn) if t >= first)
    check('Sec 4.1: LGN-KM error first below FNO at t', '33', first, label)
    check('Sec 4.1: ... and stays below to t=200 (1 = yes)', '1', float(stays), label)
    check('Sec 4.1: true enstrophy at t=11', '1.2', float(r[11]['true_enstrophy']), label)
    check('Sec 4.1: true enstrophy at t=200', '53', float(r[200]['true_enstrophy']), label)
    check('A.9: LGN-KM predicted enstrophy at t=200', '0.3', float(r[200]['lgn_enstrophy']), label)
    check('A.9: LGN-KM error above 1 at every t >= 30 (1 = yes)', '1',
          float(all(float(r[t]['lgn_rel_l2']) > 1 for t in T if t >= 30)), label)
    s = {x['model']: x['first_blowup_t (max|w|>1e3)'] for x in _rows(os.path.join(d, 'E3_summary.csv'))}
    check('Table 1 / Sec 4.1: FNO leaves physical range at t', '81', float(s['fno']), label)
    check('Sec 4.1: LRAN leaves physical range at t', '37', float(s['lran']), label)
    check('Sec 4.1: LGN-KM bounded to t=200 (1 = never blows up)', '1', float(s['lgn'] == 'never'), label)
    check('Sec 4.1 / Table 1: KNO bounded to t=200 (1 = never blows up)', '1', float(s['kno'] == 'never'), label)


def _table_A7(check, d, label):
    s = {x['metric']: x for x in _rows(os.path.join(d, 'E4_summary.csv'))}
    for metric, row in [('mean_all_frac', 'all 27 times'), ('mean_phase_0.5', 'half-integer times')]:
        paper = {'mean_all_frac': ('0.190', '0.200', '0.135'), 'mean_phase_0.5': ('0.190', '0.205', '0.159')}[metric]
        for col, pv in zip(['lgn_direct', 'interp_lgn', 'interp_true'], paper):
            check(f'Table A.7 {row}: {col}', pv, float(s[metric][col]), label)
    q = _rows(os.path.join(d, 'E4_fine_dt.csv'))
    wins = sum(float(x['lgn_direct']) < float(x['interp_lgn']) for x in q)
    check('A.9 / Sec 4.1: direct query beats own-frame interpolation (of 27)', '27', wins, label)
    margin = {ph: float(s[f'mean_phase_{ph}']['interp_lgn']) - float(s[f'mean_phase_{ph}']['lgn_direct'])
              for ph in ['0.25', '0.5', '0.75']}
    check('A.9: largest margin at half-integer offsets (1 = yes)', '1',
          float(max(margin, key=margin.get) == '0.5'), label)


def _find_long_files(data_dir):
    cands = [data_dir, os.path.join(data_dir, '..', 'long_horizon'),
             os.path.join(HERE, 'outputs', 'long_horizon'), os.environ.get('LGNKM_LONG_DIR', '')]
    for c in cands:
        a = os.path.join(c, 'ns_long_nu1e5_N200_T200_fixed.mat')
        b = os.path.join(c, 'ns_finedt_nu1e5_N200_T20_fixed.mat')
        if c and os.path.isfile(a) and os.path.isfile(b):
            return a, b
    return None


def _fig5ab(ck, check, data_dir):
    """Fig. 5a (per-step error, 200 test trajectories) and 5b (decoded enstrophy, 5 trajectories)."""
    sys.path.insert(0, os.path.join(HERE, 'navier_stokes'))
    import train_lgnkm, train_fno
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, x, y, grid, H, W = train_lgnkm.load_data(os.path.join(data_dir, 'NavierStokes_V1e-5_N1200_T20.mat'),
                                                   10, 10, 1000, 200)
    lgn = train_lgnkm.SpatialKoopmanLGN(12, 32, 32, 10, H, W).to(dev)
    lgn.load_state_dict(torch.load(f'{ck}/ns/lgnkm_nu1e5_s0.pt', map_location=dev)); lgn.eval()
    fno = train_fno.FNO2d(12, 12, 20, 10).to(dev)
    fno.load_state_dict(torch.load(f'{ck}/ns/fno_nu1e5.pt', map_location=dev)); fno.eval()
    grid = grid.to(dev)
    with torch.no_grad():
        el, ef = [], []
        for i in range(0, x.shape[0], 20):
            xb = x[i:i + 20].to(dev); yb = y[i:i + 20].to(dev).permute(0, 3, 1, 2)
            g = grid.unsqueeze(0).expand(xb.shape[0], -1, -1, -1)
            for out, model in [(el, lambda: lgn(xb, g, 10)), (ef, lambda: fno.rollout(xb, g, 10))]:
                p = model(); B = p.shape[0]
                out.append(torch.stack([((p[:, t] - yb[:, t]).reshape(B, -1).norm(dim=1) /
                                         (yb[:, t].reshape(B, -1).norm(dim=1) + 1e-8)) for t in range(10)], 1).cpu())
        el = torch.cat(el).mean(0); ef = torch.cat(ef).mean(0)
        check('Fig 5a: FNO more accurate than LGN-KM at every step 1-10 (1 = yes)', '1',
              float(bool((ef < el).all())), 'recomputed, 200 test trajectories')
        xb = x[:5].to(dev); g = grid.unsqueeze(0).expand(5, -1, -1, -1)
        ek = (lgn(xb, g, 200) ** 2).mean(dim=(-2, -1)).mean(0)
        efn = (fno.rollout(xb, g, 200) ** 2).mean(dim=(-2, -1)).mean(0)
    check('Fig 5b / Sec 4.1: FNO enstrophy at T=200, order of magnitude', '18',
          math.floor(math.log10(float(efn[-1]))), 'recomputed, 5 test trajectories')
    check('Fig 5b: LGN-KM enstrophy stays bounded, max over 200 steps < 10 (1 = yes)', '1',
          float(float(ek.max()) < 10), 'recomputed, 5 test trajectories')


def _rerun_tables(ck, check, files):
    py = sys.executable
    with tempfile.TemporaryDirectory() as tmp:
        cmd_long = [py, os.path.join(HERE, 'long_horizon', 'eval_long_horizon.py'), '--data', files[0],
                    '--koopman_ckpt', f'{ck}/ns/lgnkm_nu1e5_s0.pt', '--fno_ckpt', f'{ck}/ns/fno_nu1e5.pt',
                    '--lran_ckpt', f'{ck}/baselines/lran_nu1e5_s0.pt', '--kno_ckpt', f'{ck}/baselines/kno_nu1e5_s0.pt',
                    '--save_dir', os.path.join(tmp, 'long')]
        cmd_fd = [py, os.path.join(HERE, 'long_horizon', 'eval_fine_dt.py'), '--data', files[1],
                  '--koopman_ckpt', f'{ck}/ns/lgnkm_nu1e5_s0.pt', '--save_dir', os.path.join(tmp, 'finedt')]
        for cmd in (cmd_long, cmd_fd):
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=HERE)
        _table_A6(check, os.path.join(tmp, 'long'), 're-evaluated on regenerated data')
        _table_A7(check, os.path.join(tmp, 'finedt'), 're-evaluated on regenerated data')


def run(ck, check, check_pm, data_dir=None):
    print("\nBaselines, long horizon, continuous time (Table 1, Sec 4.1, Fig 5, A.9)")
    # --- Table 1: baselines -------------------------------------------------------
    for nu, steps, p_dmd, p_lran, p_kno in [('1e3', 40, '0.388', ('0.048', '0.001'), ('0.016', '0.001')),
                                             ('1e5', 10, '0.444', ('0.247', '0.002'), ('0.169', '0.001'))]:
        dmd = _rows(os.path.join(BR, f'dmd_results_dmd_nu{nu}_s0.csv'))
        best = min(dmd, key=lambda r: float(r['test_rel_l2']))
        check(f'Table 1 DMD nu={nu}, best rank ({steps} steps)', p_dmd, float(best['test_rel_l2']), 'shipped CSV')
        check(f'A.9 DMD nu={nu}: best rank', {'1e3': '32', '1e5': '256'}[nu], float(best['rank']), 'shipped CSV')
        lr = [_kv(os.path.join(BR, f'lran_results_lran_nu{nu}_s{s}.csv'))['test_rel_l2_full_volume'] for s in range(3)]
        kn = [_kv(os.path.join(BR, f'kno_results_kno_nu{nu}_s{s}.csv'))['test_rel_l2_full_volume'] for s in range(3)]
        check_pm(f'Table 1 LRAN nu={nu}, 3 seeds', p_lran[0], p_lran[1], lr, 'shipped CSVs')
        check_pm(f'Table 1 KNO nu={nu}, 3 seeds', p_kno[0], p_kno[1], kn, 'shipped CSVs')
    mu = [float(r['max_abs_mu']) for nu in ['1e3', '1e5'] for r in _rows(os.path.join(BR, f'dmd_results_dmd_nu{nu}_s0.csv'))]
    check('Table 1 DMD max|mu|, minimum over ranks and nu', '1.01', min(mu), 'shipped CSVs')
    check('Table 1 DMD max|mu|, maximum over ranks and nu', '1.07', max(mu), 'shipped CSVs')
    rad = {nu: [float(np.abs(np.linalg.eigvals(torch.load(f'{ck}/baselines/lran_nu{nu}_s{s}.pt',
                map_location='cpu')['A'].double().numpy())).max()) for s in range(3)] for nu in ['1e3', '1e5']}
    for s in range(3):
        check(f'Table 1 / A.9 LRAN spectral radius nu=1e-3, seed {s}', '1.05', rad['1e3'][s], 'recomputed, checkpoint')
    check('Table 1 / A.9 LRAN spectral radius nu=1e-5, minimum of 3 seeds', '1.33', min(rad['1e5']), 'recomputed, checkpoint')
    check('Table 1 / A.9 LRAN spectral radius nu=1e-5, maximum of 3 seeds', '1.39', max(rad['1e5']), 'recomputed, checkpoint')

    # --- A.9 / Sec 4.1: long horizon and fine dt (shipped Nov 3 2026 results) -------
    _table_A6(check, os.path.join(LH, 'long'), 'shipped Nov 3 CSV')
    _table_A7(check, os.path.join(LH, 'finedt'), 'shipped Nov 3 CSV')
    for f, pv in [('ns_long_nu1e5_N200_T200_fixed_gate.json', '0.95'), ('ns_finedt_nu1e5_N200_T20_fixed_gate.json', '0.96')]:
        g = json.load(open(os.path.join(LH, f)))
        check(f'A.9 initial enstrophy / benchmark ({f.split("_")[1]})', pv, g['ratio'], 'shipped gate JSON')
        check(f'A.9 trajectories ({f.split("_")[1]})', '200', g['n'], 'shipped gate JSON')

    # --- Fig. 5 -----------------------------------------------------------------------
    e = _rows(os.path.join(LH, 'fig5ab', 'long_horizon_energy.csv'))
    check('Fig 5b / Sec 4.1: FNO enstrophy at T=200, order of magnitude', '18',
          math.floor(math.log10(float(e[-1]['fno_energy']))), 'shipped CSV')
    ps = _rows(os.path.join(LH, 'fig5ab', 'long_horizon_per_step.csv'))
    check('Fig 5a: FNO more accurate than LGN-KM at every step 1-10 (1 = yes)', '1',
          float(all(float(r['fno_l2_mean']) < float(r['koopman_l2_mean']) for r in ps)), 'shipped CSV')
    c = {int(r['horizon_T']): r for r in _rows(os.path.join(HERE, 'continuous_time', 'results',
                                                            'continuous_time_cost_scaling.csv'))}
    check('Fig 5c / Sec 4.1: LGN-KM wall time at T=1 (ms)', '1.7', float(c[1]['koopman_ms']), 'shipped timing CSV')
    check('Fig 5c / Sec 4.1: LGN-KM wall time at T=200 (ms)', '1.9', float(c[200]['koopman_ms']), 'shipped timing CSV')
    check('Fig 5c / Sec 4.1: FNO wall time at T=200 (ms)', '299', float(c[200]['fno_ms']), 'shipped timing CSV')

    if data_dir:
        _fig5ab(ck, check, data_dir)
        files = _find_long_files(data_dir)
        if files:
            _rerun_tables(ck, check, files)
        else:
            print("  (regenerated long-horizon files not found; Tables A.6/A.7 checked from the shipped CSVs only)")
