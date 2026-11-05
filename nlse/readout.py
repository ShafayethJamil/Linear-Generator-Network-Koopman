"""Rotation rates of S, routing of optical frequencies onto them, and the beta2 readout.

Paper: Section 3.3, Fig. 4, Appendix A.8.

Usage:
    python nlse/readout.py --ckpt checkpoints/nlse/b2_20_s1.pt --data data/nlse/nlse_b2_20_soliton.mat --beta2 -20 [--out readout.json]

Reports the 16 rotation rates of S (radians per snapshot), the rotation rate carrying most of each optical
mode's latent energy on the test pulses (1000:1200), its through-origin slope and rank correlation against the
predicted phase advance 0.5 |beta2| w_k^2 dz (modes k > 0 with >= 1% of the energy), and
|beta2| = 13.56 x (top rate) at dz = 0.166 km. The beta2 readout needs only the checkpoint.
"""
import sys, os, json, argparse, warnings, contextlib, io
import numpy as np, torch, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_nlse as M
torch.set_num_threads(4)
OMEGA = 2 * np.pi * np.arange(16) / 100.0     # rad/ps, 100 ps window, rfft modes 0..15
DZ_DEFAULT = 0.166                             # km per snapshot (5 km / 30 saved intervals; last interval 20 m)


def spearman(a, b):
    ra = pd.Series(a).rank().values; rb = pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def analyse(ckpt, data, beta2, dz=DZ_DEFAULT, curve=None, curve_stat='final'):
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, x_te, _, grid, N_t, outc = M.load_nlse_data(data, Z_in=10, Z_out=10, n_train=1000, n_test=200, field='complex')
    sd = torch.load(ckpt, map_location='cpu')
    model = M.KoopmanLGN_1D(modes=16, width=32, r=32, Z_in=x_te.shape[-1], N_t=N_t, out_channels=outc)
    model.load_state_dict(sd); model.eval()
    with torch.no_grad():
        g = grid.unsqueeze(0).expand(x_te.shape[0], -1, -1)
        c = torch.fft.rfft(model.encoder(x_te, g))[:, :, :16].to(torch.complex128)
    S = ((model.P - model.P.T) / 2).double().detach()
    ev, V = torch.linalg.eig(S)
    absw = np.abs(ev.imag.numpy()); Vn = V.numpy()
    rungs = np.sort(absw)[::2]
    phi = 0.5 * abs(beta2) * OMEGA ** 2 * dz
    E = np.zeros((16, 32))
    for k in range(16):
        a = np.conj(Vn.T) @ c[:, :, k].numpy().T
        E[k] = (np.abs(a) ** 2).sum(1)
    share = E.sum(1) / E.sum()
    f_top, f_wtd = [], []
    for k in range(16):
        e = E[k]; pairs = {}
        for j in range(32):
            key = round(absw[j], 6); pairs[key] = pairs.get(key, 0) + e[j]
        f_top.append(max(pairs, key=pairs.get)); f_wtd.append(float((e * absw).sum() / e.sum()))
    f_top = np.array(f_top); f_wtd = np.array(f_wtd)
    sel = share >= 0.01
    sel[0] = False  # k=0 has phi=0; excluded from slope/Spearman
    slope = float((f_top[sel] @ phi[sel]) / (phi[sel] @ phi[sel]))
    rho = spearman(f_top[sel], phi[sel])
    top = float(rungs[-1]); phi15 = float(phi[15])
    out = dict(ckpt=ckpt, data=data, beta2=beta2, dz=dz, rungs=rungs.tolist(),
               omega=OMEGA.tolist(), phi=phi.tolist(), f_top=f_top.tolist(), f_wtd=f_wtd.tolist(), share=share.tolist(),
               link1=dict(n_modes=int(sel.sum()), spearman=rho, slope=slope),
               link2=dict(top_rung=top, phi15=phi15, ratio=top / phi15, beta2_implied=top / (0.5 * OMEGA[15] ** 2 * dz),
                          const_derived=1 / (0.5 * OMEGA[15] ** 2 * dz)))
    if curve and os.path.exists(curve):
        cv = pd.read_csv(curve)
        out['test_l2'] = float(cv.test_l2.iloc[-1] if curve_stat == 'final' else cv.test_l2.min())
        out['test_l2_stat'] = curve_stat
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data', required=True, help='the dataset the model was trained on (test pulses 1000:1200)')
    p.add_argument('--beta2', type=float, required=True, help='true beta2 in ps^2/km, e.g. -20')
    p.add_argument('--dz', type=float, default=DZ_DEFAULT, help='km per snapshot')
    p.add_argument('--out', default=None, help='optional JSON output path')
    a = p.parse_args()
    r = analyse(a.ckpt, a.data, a.beta2, dz=a.dz)
    l1, l2 = r['link1'], r['link2']
    print(f"routing: slope {l1['slope']:.3f}, rank correlation {l1['spearman']:.3f} over {l1['n_modes']} modes")
    print(f"top rate {l2['top_rung']:.4f} rad/snapshot = {l2['ratio']:.3f} x predicted band-edge advance; "
          f"|beta2| read as {l2['const_derived']:.2f} x top = {l2['beta2_implied']:.2f} ps^2/km (true {abs(a.beta2):g})")
    if a.out:
        json.dump(r, open(a.out, 'w'), indent=1)
