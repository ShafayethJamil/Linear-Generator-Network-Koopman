"""Compare FitzHugh-Nagumo generators trained at two diffusivities.

Paper: Appendix A.4, Fig. A.2c.

Usage:
    python fitzhugh_nagumo/analyze_fhn_universality.py --ckpt_Du1 checkpoints/fhn/periodic_Du0.01_s0.pt \
        --ckpt_Du2 checkpoints/fhn/periodic_Du0.001_s0.pt

Reports the raw S cosine similarity and R^2 on [0,1]-normalised profiles of the S singular
values, sorted d and sorted alpha, with the D_u = 0.001 model as the reference.
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse, os, csv


def extract(ckpt_path, label):
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    P = state['P']
    d_raw = state['d']
    alpha_raw = state['alpha']

    r = P.shape[0]
    S = (P - P.T) / 2.0
    d0 = F.softplus(d_raw)
    a0 = F.softplus(alpha_raw)

    # SVD of S
    svd = torch.linalg.svdvals(S)

    print(f"[{label}]")
    print(f"  ||S||_F = {S.norm().item():.4f}")
    print(f"  d0:  [{d0.min().item():.6f}, {d0.max().item():.6f}], mean={d0.mean().item():.6f}")
    print(f"  a0:  [{a0.min().item():.6f}, {a0.max().item():.6f}], mean={a0.mean().item():.6f}")

    return S, d0, a0, svd


def normalized_r2(x, y):
    """R² on [0,1]-normalized profiles (same as paper Section 3.2)."""
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-10)
    ss_res = ((x_n - y_n)**2).sum().item()
    ss_tot = ((y_n - y_n.mean())**2).sum().item()
    return 1 - ss_res / (ss_tot + 1e-10)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("FHN Cross-Diffusion Universality Analysis")
    print("=" * 60)

    S1, d01, a01, svd1 = extract(args.ckpt_Du1, 'Du=0.01')
    S2, d02, a02, svd2 = extract(args.ckpt_Du2, 'Du=0.001')

    # --- Raw S cosine similarity ---
    cos_sim = (S1.flatten() @ S2.flatten()) / (S1.flatten().norm() * S2.flatten().norm() + 1e-8)

    # --- Singular-value profile of S ---
    r2_svd = normalized_r2(svd1, svd2)

    # --- Sorted d0 distribution ---
    d01_sorted = d01.sort().values
    d02_sorted = d02.sort().values
    r2_d = normalized_r2(d01_sorted, d02_sorted)

    # --- Sorted alpha distribution ---
    a01_sorted = a01.sort().values
    a02_sorted = a02.sort().values
    r2_alpha = normalized_r2(a01_sorted, a02_sorted)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Raw S cosine similarity:       {cos_sim.item():.3f}")
    print(f"  S SVD profile match (R²):      {r2_svd:.3f}")
    print(f"  Sorted d₀ match (R²):          {r2_d:.3f}")
    print(f"  Sorted α match (R²):           {r2_alpha:.3f}")

    # Save CSV
    csv_path = os.path.join(args.out_dir, 'fhn_universality.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['S_cosine_similarity', f'{cos_sim.item():.4f}'])
        w.writerow(['S_SVD_profile_R2', f'{r2_svd:.4f}'])
        w.writerow(['sorted_d0_R2', f'{r2_d:.4f}'])
        w.writerow(['sorted_alpha_R2', f'{r2_alpha:.4f}'])

    # Save sorted profiles for plotting
    np.savetxt(os.path.join(args.out_dir, 'svd_Du0.01.csv'),
               svd1.numpy(), delimiter=',', fmt='%.8f')
    np.savetxt(os.path.join(args.out_dir, 'svd_Du0.001.csv'),
               svd2.numpy(), delimiter=',', fmt='%.8f')
    np.savetxt(os.path.join(args.out_dir, 'd0_sorted_Du0.01.csv'),
               d01_sorted.numpy(), delimiter=',', fmt='%.8f')
    np.savetxt(os.path.join(args.out_dir, 'd0_sorted_Du0.001.csv'),
               d02_sorted.numpy(), delimiter=',', fmt='%.8f')
    np.savetxt(os.path.join(args.out_dir, 'alpha_sorted_Du0.01.csv'),
               a01_sorted.numpy(), delimiter=',', fmt='%.8f')
    np.savetxt(os.path.join(args.out_dir, 'alpha_sorted_Du0.001.csv'),
               a02_sorted.numpy(), delimiter=',', fmt='%.8f')

    print(f"\n  Saved to {args.out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_Du1', required=True)
    parser.add_argument('--ckpt_Du2', required=True)
    parser.add_argument('--out_dir', default='outputs/fhn_universality')
    args = parser.parse_args()
    main(args)
