"""
Fig. 6(b,c) and Table A.1 numbers from the transfer learning curves.

No script producing the Fig. 6(c) metrics survived from the original runs; this one
implements the definitions that reproduce the plotted values:

  best(cond, N)  = minimum test L2 over the 500 epochs of curve_<cond>_N<N>_s0.csv
  speedup(N)     = (epoch of the scratch minimum)
                   / (first epoch at which freeze-S test L2 <= the scratch minimum)
                   (undefined when freeze-S never reaches the scratch minimum)
  gain(N)        = (best(scratch) - best(freeze-S)) / best(scratch)
  Delta(N)       = (best(freeze-S) - best(transfer-all)) / best(transfer-all),
                   from unrounded errors (Table A.1)

All values are single-seed (seed 0).

Usage (from the repo root):
    python transfer/fig6_metrics.py --results_dir transfer/results [--out fig6_metrics.csv]
"""
import argparse, csv, os

BUDGETS = [50, 100, 200, 500, 1000]


def load_curve(results_dir, cond, n, seed=0):
    with open(os.path.join(results_dir, f'curve_{cond}_N{n}_s{seed}.csv')) as f:
        rows = list(csv.DictReader(f))
    return [int(r['epoch']) for r in rows], [float(r['test_l2']) for r in rows]


def metrics(results_dir):
    out = []
    for n in BUDGETS:
        ep_s, l2_s = load_curve(results_dir, 'scratch', n)
        ep_f, l2_f = load_curve(results_dir, 'transfer_freezeS', n)
        _, l2_a = load_curve(results_dir, 'transfer_all', n)
        best_s, best_f, best_a = min(l2_s), min(l2_f), min(l2_a)
        ep_best_s = ep_s[l2_s.index(best_s)]
        reach = next((e for e, v in zip(ep_f, l2_f) if v <= best_s), None)
        out.append(dict(N=n, scratch=best_s, freezeS=best_f, transfer_all=best_a,
                        scratch_best_epoch=ep_best_s, freezeS_reaches_scratch_best=reach,
                        speedup=(ep_best_s / reach) if reach else float('nan'),
                        gain_pct=100 * (best_s - best_f) / best_s,
                        gain_all_pct=100 * (best_s - best_a) / best_s,
                        delta_pct=100 * (best_f - best_a) / best_a))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='transfer/results')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    m = metrics(a.results_dir)
    print(f"{'N':>5} {'scratch':>8} {'freezeS':>8} {'all':>8} {'speedup':>8} {'gain%':>7} {'Delta%':>7}")
    for r in m:
        print(f"{r['N']:>5} {r['scratch']:8.4f} {r['freezeS']:8.4f} {r['transfer_all']:8.4f} "
              f"{r['speedup']:8.2f} {r['gain_pct']:7.1f} {r['delta_pct']:7.2f}")
    if a.out:
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(m[0])); w.writeheader(); w.writerows(m)
