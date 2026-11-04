"""
Extract S, D, alpha, d from Koopman checkpoints into CSV files.
Run on your GPU machine, then upload the CSVs.

Usage:
    python extract_SD.py \
        --ckpt_1e5 checkpoints/best_koopman_nu1e5_raw.pt \
        --ckpt_1e3 checkpoints/best_koopman_nu1e3_raw.pt \
        --out_dir sd_analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse, os, csv

def extract(ckpt_path, label, out_dir):
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    P = state['P']           # (r, r)
    d_raw = state['d']       # (r,)
    alpha_raw = state['alpha']  # (r,)
    k2_norm = state['k2_norm']  # (modes, modes)

    r = P.shape[0]
    modes = k2_norm.shape[0]

    # Derived quantities
    S = (P - P.T) / 2.0       # (r, r) skew-symmetric
    d0 = F.softplus(d_raw)     # (r,)
    a0 = F.softplus(alpha_raw) # (r,)

    # Save S matrix
    np.savetxt(os.path.join(out_dir, f'S_{label}.csv'),
               S.numpy(), delimiter=',', fmt='%.8f')

    # Save P matrix (raw)
    np.savetxt(os.path.join(out_dir, f'P_{label}.csv'),
               P.numpy(), delimiter=',', fmt='%.8f')

    # Save d0 and a0 (per-channel)
    with open(os.path.join(out_dir, f'channels_{label}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['channel', 'd_raw', 'd0_softplus', 'alpha_raw', 'alpha_softplus'])
        for i in range(r):
            w.writerow([i, f'{d_raw[i].item():.8f}', f'{d0[i].item():.8f}',
                       f'{alpha_raw[i].item():.8f}', f'{a0[i].item():.8f}'])

    # Save eigenspectrum of L_k for all modes
    with open(os.path.join(out_dir, f'eigenspectrum_{label}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['k1', 'k2', 'k2_mag', 'k2_norm_val',
                     'dom_re', 'dom_im', 'all_re_min', 'all_re_max',
                     'D_diag_mean', 'D_diag_min', 'D_diag_max'])
        for i in range(modes):
            for j in range(modes):
                D_diag = d0 + a0 * k2_norm[i, j]
                L = S - torch.diag(D_diag)
                eigs = torch.linalg.eigvals(L)
                dom_idx = eigs.real.argmax()
                w.writerow([i, j, i**2 + j**2, f'{k2_norm[i,j].item():.6f}',
                           f'{eigs[dom_idx].real.item():.8f}',
                           f'{eigs[dom_idx].imag.item():.8f}',
                           f'{eigs.real.min().item():.8f}',
                           f'{eigs.real.max().item():.8f}',
                           f'{D_diag.mean().item():.8f}',
                           f'{D_diag.min().item():.8f}',
                           f'{D_diag.max().item():.8f}'])

    # Save singular values of S (characterize energy transfer strength)
    svd = torch.linalg.svdvals(S)
    with open(os.path.join(out_dir, f'S_svd_{label}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['index', 'singular_value'])
        for i, sv in enumerate(svd):
            w.writerow([i, f'{sv.item():.8f}'])

    print(f"[{label}] Extracted: r={r}, modes={modes}")
    print(f"  S norm (Frobenius): {S.norm().item():.6f}")
    print(f"  d0 range: [{d0.min().item():.6f}, {d0.max().item():.6f}]")
    print(f"  a0 range: [{a0.min().item():.6f}, {a0.max().item():.6f}]")
    print(f"  a0 mean:  {a0.mean().item():.6f}")

    return S, d0, a0


def compare(S1, d01, a01, S2, d02, a02, out_dir):
    """Direct comparison metrics."""
    with open(os.path.join(out_dir, 'comparison.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'nu1e5', 'nu1e3', 'ratio_or_similarity'])

        # S similarity: cosine similarity of flattened S
        s1f = S1.flatten()
        s2f = S2.flatten()
        cos_sim = (s1f @ s2f) / (s1f.norm() * s2f.norm() + 1e-8)
        w.writerow(['S_cosine_similarity', '', '', f'{cos_sim.item():.6f}'])
        w.writerow(['S_frobenius_norm', f'{S1.norm().item():.6f}',
                    f'{S2.norm().item():.6f}',
                    f'{S1.norm().item()/S2.norm().item():.6f}'])

        # Element-wise correlation of S
        corr = np.corrcoef(S1.numpy().flatten(), S2.numpy().flatten())[0, 1]
        w.writerow(['S_pearson_correlation', '', '', f'{corr:.6f}'])

        # Frobenius distance
        w.writerow(['S_frobenius_distance', '', '',
                    f'{(S1-S2).norm().item():.6f}'])

        # d0 comparison
        d_corr = np.corrcoef(d01.numpy(), d02.numpy())[0, 1]
        w.writerow(['d0_mean', f'{d01.mean().item():.6f}', f'{d02.mean().item():.6f}',
                    f'{d01.mean().item()/d02.mean().item():.6f}'])
        w.writerow(['d0_pearson_correlation', '', '', f'{d_corr:.6f}'])

        # alpha comparison
        a_corr = np.corrcoef(a01.numpy(), a02.numpy())[0, 1]
        w.writerow(['a0_mean', f'{a01.mean().item():.6f}', f'{a02.mean().item():.6f}',
                    f'{a01.mean().item()/a02.mean().item():.6f}'])
        w.writerow(['a0_pearson_correlation', '', '', f'{a_corr:.6f}'])

        # Channel-by-channel D comparison at k=(0,0) and k=(11,11)
        D_00_1 = d01  # D at k=0
        D_00_2 = d02
        D_max_1 = d01 + a01  # D at k=kmax (k2_norm=1)  
        D_max_2 = d02 + a02
        w.writerow(['D_k0_mean', f'{D_00_1.mean().item():.6f}',
                    f'{D_00_2.mean().item():.6f}',
                    f'{D_00_1.mean().item()/D_00_2.mean().item():.6f}'])
        w.writerow(['D_kmax_mean', f'{D_max_1.mean().item():.6f}',
                    f'{D_max_2.mean().item():.6f}',
                    f'{D_max_1.mean().item()/D_max_2.mean().item():.6f}'])

    print(f"\nComparison saved to {out_dir}/comparison.csv")
    print(f"  S cosine similarity:      {cos_sim.item():.4f}")
    print(f"  S Pearson correlation:     {corr:.4f}")
    print(f"  S Frobenius distance:      {(S1-S2).norm().item():.4f}")
    print(f"  d0 correlation:            {d_corr:.4f}")
    print(f"  alpha correlation:         {a_corr:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_1e5', required=True)
    parser.add_argument('--ckpt_1e3', required=True)
    parser.add_argument('--out_dir',  default='./sd_analysis')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Extracting ν=1e-5...")
    S1, d01, a01 = extract(args.ckpt_1e5, 'nu1e5', args.out_dir)

    print("\nExtracting ν=1e-3...")
    S2, d02, a02 = extract(args.ckpt_1e3, 'nu1e3', args.out_dir)

    print("\nComparing...")
    compare(S1, d01, a01, S2, d02, a02, args.out_dir)

    print(f"\nAll files in {args.out_dir}/:")
    for f in sorted(os.listdir(args.out_dir)):
        print(f"  {f}")
