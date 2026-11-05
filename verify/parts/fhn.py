"""FitzHugh-Nagumo checks (Appendix A.4 periodic, Appendix A.7 / Table A.4 no-flux).

run(ck, check, check_pm, data_dir=None) is called by verify.py. Expected values are
the numbers as printed in the paper. Checkpoint-only checks need nothing else;
the Table A.4 test errors need the no-flux datasets in data_dir
(rd_neumann_Du0.01_N1200_T30.mat, rd_neumann_Du0.001_N1200_T30.mat).
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def _parts(sd):
    P = sd['P'].double()
    return ((P - P.T) / 2), F.softplus(sd['d'].double()), F.softplus(sd['alpha'].double()), sd['k2_norm'].double()


def _eigs(sd):
    """All eigenvalues of L_k = S - diag(d + alpha k2_norm) over the retained modes."""
    S, d0, a0, k2n = _parts(sd)
    M = k2n.shape[0]
    return np.array([torch.linalg.eigvals(S - torch.diag(d0 + a0 * k2n[i, j])).numpy()
                     for i in range(M) for j in range(M)]), M


def _dominant_r2(sd):
    """fitzhugh_nagumo/analyze_fhn_spectrum.py: Re(lambda_dom) vs |k|^2 (integer i^2+j^2), k > 0, OLS R^2."""
    e, M = _eigs(sd)
    k2 = np.array([i * i + j * j for i in range(M) for j in range(M)], dtype=float)
    dom = e.real.max(axis=1)
    mask = k2 > 0
    c = np.polyfit(k2[mask], dom[mask], 1)
    res = dom[mask] - np.polyval(c, k2[mask])
    return 1 - (res ** 2).sum() / ((dom[mask] - dom[mask].mean()) ** 2).sum()


def _r2_norm(x, y):
    """fitzhugh_nagumo/analyze_fhn_universality.py normalized_r2: [0,1]-normalised x against reference y."""
    xn = (x - x.min()) / (x.max() - x.min() + 1e-10)
    yn = (y - y.min()) / (y.max() - y.min() + 1e-10)
    return float(1 - ((xn - yn) ** 2).sum() / (((yn - yn.mean()) ** 2).sum() + 1e-10))


def run(ck, check, check_pm, data_dir=None):
    print("\nFitzHugh-Nagumo, periodic (Appendix A.4, Fig. A.2)")
    p1, p2 = _load(f'{ck}/fhn/periodic_Du0.01_s0.pt'), _load(f'{ck}/fhn/periodic_Du0.001_s0.pt')
    check('dominant-branch R^2, Re(lambda) vs |k|^2, Du=0.01', '0.950', _dominant_r2(p1), 'A.4')
    check('dominant-branch R^2, Re(lambda) vs |k|^2, Du=0.001', '0.940', _dominant_r2(p2), 'A.4')
    e1, _ = _eigs(p1)
    check('eigenvalues with Re(lambda) < 0, Du=0.01 (of 4,608)', '4608', int((e1.real < 0).sum()), 'A.4')
    S1, d1, a1, _ = _parts(p1); S2, d2, a2, _ = _parts(p2)
    cos = float((S1.flatten() @ S2.flatten()) / (S1.norm() * S2.norm()))
    check('raw S cosine similarity, Du=0.01 vs 0.001', '0.449', cos, 'A.4 (both seed 0)')
    check('S singular-value profile R^2', '0.994',
          _r2_norm(torch.linalg.svdvals(S1).numpy(), torch.linalg.svdvals(S2).numpy()), 'A.4, Sec 3.2')
    check('sorted base damping d R^2', '0.945', _r2_norm(np.sort(d1.numpy()), np.sort(d2.numpy())), 'A.4')
    check('sorted Laplacian coefficient alpha R^2', '0.820', _r2_norm(np.sort(a1.numpy()), np.sort(a2.numpy())), 'A.4')

    print("\nFitzHugh-Nagumo, no-flux boundaries (Appendix A.7, Table A.4)")
    slope = {}
    for du, paper_maxre in [('0.01', '-0.084'), ('0.001', '-0.082')]:
        mx, slopes, n_stable = [], [], 0
        for s in range(3):
            sd = _load(f'{ck}/fhn/neumann_Du{du}_s{s}.pt')
            e, M = _eigs(sd)
            mx.append(float(e.real.max())); n_stable += int((e.real < 0).sum())
            _, _, a0, _ = _parts(sd)
            slopes.append(float(a0.mean()))          # mean decay vs k2_norm is exactly d.mean + alpha.mean * k2_norm
        check(f'max Re(lambda), Du={du} (mean of 3 seeds)', paper_maxre, float(np.mean(mx)), 'Table A.4')
        check(f'stable eigenvalues, Du={du}, 3 seeds (of 13,824)', '13824', n_stable, 'A.7 "every eigenvalue stable"')
        slope[du] = float(np.mean(slopes))
    check('dissipation slope larger at the larger diffusivity (1 = yes)', '1',
          1.0 if slope['0.01'] > slope['0.001'] else 0.0, 'A.7')

    if data_dir:
        sys.path.insert(0, os.path.join(REPO, 'fitzhugh_nagumo'))
        import train_fhn_neumann as T
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for du, paper in [('0.01', ['0.0017', '0.0017', '0.0017']), ('0.001', ['0.0033', '0.0034', '0.0035'])]:
            path = os.path.join(data_dir, f'rd_neumann_Du{du}_N1200_T30.mat')
            if not os.path.isfile(path):
                print(f"  (skipped Table A.4 test L2 at Du={du}: {path} not found)")
                continue
            _, _, x, y, grid, H, W = T.load_data(path, 10, 20, 1000, 200)
            for s in range(3):
                m = T.SpatialKoopmanLGN_DCT(12, 32, 32, 10, H, W).to(dev)
                m.load_state_dict(_load(f'{ck}/fhn/neumann_Du{du}_s{s}.pt')); m.eval()
                err = 0.0
                with torch.no_grad():
                    for i in range(0, x.shape[0], 10):
                        xb, yb = x[i:i + 10].to(dev), y[i:i + 10].to(dev).permute(0, 3, 1, 2)
                        g = grid.to(dev).unsqueeze(0).expand(xb.shape[0], -1, -1, -1)
                        err += T.paper_rel_l2_loss(m(xb, g, 20), yb).item() * xb.shape[0]
                check(f'no-flux test rel. L2, Du={du}, seed {s}', paper[s], err / x.shape[0], 'Table A.4')
