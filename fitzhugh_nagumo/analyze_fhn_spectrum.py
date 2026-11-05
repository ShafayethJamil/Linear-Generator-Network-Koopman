"""
Eigenspectrum of a trained FHN generator (Appendix A.4, Fig. A.2a-b)
=====================================================================
Writes the full eigenspectrum, the dominant (least-damped) eigenvalue per
wavevector, and the R^2 of Re(lambda_dom) vs |k|^2 over all k > 0
(pde_fingerprint_comparison.csv, column R2_Re_vs_k2). Output files keep the
original script's 'swe' label.

Usage:
    python fitzhugh_nagumo/analyze_fhn_spectrum.py \
        --ckpt_swe checkpoints/fhn/periodic_Du0.01_s0.pt --out_dir results/fhn_Du0.01
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse, os, csv


def extract_generator(ckpt_path, label):
    """Extract S, D parameters and full eigenspectrum from checkpoint."""
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    P = state['P']
    d_raw = state['d']
    alpha_raw = state['alpha']
    k2_norm = state['k2_norm']

    r = P.shape[0]
    modes = k2_norm.shape[0]

    S = (P - P.T) / 2.0
    d0 = F.softplus(d_raw)
    a0 = F.softplus(alpha_raw)

    # Frobenius norms
    S_norm = S.norm().item()

    # Mean D across all modes
    D_all = []
    for i in range(modes):
        for j in range(modes):
            D_diag = d0 + a0 * k2_norm[i, j]
            D_all.append(D_diag.mean().item())
    D_mean = np.mean(D_all)

    # S/D ratio
    # Use Frobenius norm of S vs mean trace of D across modes
    D_trace_mean = 0.0
    for i in range(modes):
        for j in range(modes):
            D_diag = d0 + a0 * k2_norm[i, j]
            D_trace_mean += torch.diag(D_diag).norm().item()  # Frobenius of diagonal = L2 of diag
    D_trace_mean /= (modes * modes)
    SD_ratio = S_norm / D_trace_mean

    print(f"[{label}] r={r}, modes={modes}")
    print(f"  ||S||_F = {S_norm:.4f}")
    print(f"  mean ||D||_F = {D_trace_mean:.4f}")
    print(f"  S/D ratio = {SD_ratio:.4f}")
    print(f"  d0 range: [{d0.min().item():.6f}, {d0.max().item():.6f}]")
    print(f"  a0 range: [{a0.min().item():.6f}, {a0.max().item():.6f}]")

    return {
        'S': S, 'd0': d0, 'a0': a0, 'k2_norm': k2_norm,
        'r': r, 'modes': modes,
        'S_norm': S_norm, 'D_norm_mean': D_trace_mean, 'SD_ratio': SD_ratio
    }


def save_eigenspectrum(info, label, out_dir):
    """Save full eigenspectrum CSV (same format as NS)."""
    S, d0, a0, k2_norm = info['S'], info['d0'], info['a0'], info['k2_norm']
    r, modes = info['r'], info['modes']

    csv_path = os.path.join(out_dir, f'eigenspectrum_{label}.csv')
    with open(csv_path, 'w', newline='') as f:
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
    print(f"  Saved {csv_path}")


def save_full_eigenspectrum(info, label, out_dir):
    """Save ALL eigenvalues (not just dominant) for complex plane plot."""
    S, d0, a0, k2_norm = info['S'], info['d0'], info['a0'], info['k2_norm']
    r, modes = info['r'], info['modes']

    csv_path = os.path.join(out_dir, f'eigenspectrum_full_{label}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['k1', 'k2', 'k2_mag', 'k_mag', 'branch', 'eig_re', 'eig_im'])

        for i in range(modes):
            for j in range(modes):
                D_diag = d0 + a0 * k2_norm[i, j]
                L = S - torch.diag(D_diag)
                eigs = torch.linalg.eigvals(L)
                k_mag = np.sqrt(i**2 + j**2)
                # Sort by imaginary part for consistent branch assignment
                sorted_idx = eigs.imag.argsort()
                for branch, idx in enumerate(sorted_idx):
                    w.writerow([i, j, i**2 + j**2, f'{k_mag:.4f}',
                               branch,
                               f'{eigs[idx].real.item():.8f}',
                               f'{eigs[idx].imag.item():.8f}'])
    print(f"  Saved {csv_path}")


def analyze_dispersion(info, label, out_dir):
    """
    Check if Im(λ) scales with |k| (wave dispersion) vs |k|² (viscous).
    For SWE: expect Im(λ) ~ |k|  (gravity waves)
    For NS:  expect Re(λ) ~ |k|² (viscous dissipation)
    """
    S, d0, a0, k2_norm = info['S'], info['d0'], info['a0'], info['k2_norm']
    modes = info['modes']

    k_mags = []
    dom_im = []
    dom_re = []
    max_im = []  # max |Im(λ)| at each k

    for i in range(modes):
        for j in range(modes):
            k_mag = np.sqrt(i**2 + j**2)
            D_diag = d0 + a0 * k2_norm[i, j]
            L = S - torch.diag(D_diag)
            eigs = torch.linalg.eigvals(L)

            k_mags.append(k_mag)
            dom_idx = eigs.real.argmax()
            dom_re.append(eigs[dom_idx].real.item())
            dom_im.append(eigs[dom_idx].imag.item())
            max_im.append(eigs.imag.abs().max().item())

    k_mags = np.array(k_mags)
    dom_re = np.array(dom_re)
    max_im = np.array(max_im)

    # Fit Im(λ) vs |k| (wave) and vs |k|² (viscous)
    mask = k_mags > 0  # exclude k=0

    # Linear fit: max|Im(λ)| = a * |k| + b
    from numpy.polynomial import polynomial as P
    coeffs_lin = np.polyfit(k_mags[mask], max_im[mask], 1)
    pred_lin = np.polyval(coeffs_lin, k_mags[mask])
    ss_res_lin = np.sum((max_im[mask] - pred_lin)**2)
    ss_tot = np.sum((max_im[mask] - max_im[mask].mean())**2)
    r2_im_vs_k = 1 - ss_res_lin / ss_tot if ss_tot > 0 else 0

    # Quadratic fit: max|Im(λ)| = a * |k|² + b
    coeffs_quad = np.polyfit(k_mags[mask]**2, max_im[mask], 1)
    pred_quad = np.polyval(coeffs_quad, k_mags[mask]**2)
    ss_res_quad = np.sum((max_im[mask] - pred_quad)**2)
    r2_im_vs_k2 = 1 - ss_res_quad / ss_tot if ss_tot > 0 else 0

    # Re(λ) vs |k|² (viscous dissipation)
    coeffs_re = np.polyfit(k_mags[mask]**2, dom_re[mask], 1)
    pred_re = np.polyval(coeffs_re, k_mags[mask]**2)
    ss_res_re = np.sum((dom_re[mask] - pred_re)**2)
    ss_tot_re = np.sum((dom_re[mask] - dom_re[mask].mean())**2)
    r2_re_vs_k2 = 1 - ss_res_re / ss_tot_re if ss_tot_re > 0 else 0

    print(f"\n  Dispersion analysis [{label}]:")
    print(f"    max|Im(λ)| vs |k|   R² = {r2_im_vs_k:.4f}  (wave dispersion)")
    print(f"    max|Im(λ)| vs |k|²  R² = {r2_im_vs_k2:.4f}  (quadratic)")
    print(f"    Re(λ_dom)  vs |k|²  R² = {r2_re_vs_k2:.4f}  (viscous dissipation)")

    # Save dispersion data
    csv_path = os.path.join(out_dir, f'dispersion_{label}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['k_mag', 'k2_mag', 'dom_re', 'dom_im', 'max_abs_im'])
        for i in range(len(k_mags)):
            w.writerow([f'{k_mags[i]:.4f}', f'{k_mags[i]**2:.4f}',
                        f'{dom_re[i]:.8f}', f'{dom_im[i]:.8f}',
                        f'{max_im[i]:.8f}'])
    print(f"    Saved {csv_path}")

    return {
        'r2_im_vs_k': r2_im_vs_k,
        'r2_im_vs_k2': r2_im_vs_k2,
        'r2_re_vs_k2': r2_re_vs_k2
    }


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # Extract SWE
    print("=" * 60)
    print("Extracting SWE generator...")
    swe = extract_generator(args.ckpt_swe, 'swe')
    save_eigenspectrum(swe, 'swe', args.out_dir)
    save_full_eigenspectrum(swe, 'swe', args.out_dir)
    disp_swe = analyze_dispersion(swe, 'swe', args.out_dir)

    # Extract NS (if provided)
    ns_results = {}
    if args.ckpt_ns_1e5:
        print("\n" + "=" * 60)
        print("Extracting NS ν=1e-5 generator...")
        ns5 = extract_generator(args.ckpt_ns_1e5, 'ns_nu1e5')
        disp_ns5 = analyze_dispersion(ns5, 'ns_nu1e5', args.out_dir)
        ns_results['nu1e5'] = (ns5, disp_ns5)

    if args.ckpt_ns_1e3:
        print("\n" + "=" * 60)
        print("Extracting NS ν=1e-3 generator...")
        ns3 = extract_generator(args.ckpt_ns_1e3, 'ns_nu1e3')
        disp_ns3 = analyze_dispersion(ns3, 'ns_nu1e3', args.out_dir)
        ns_results['nu1e3'] = (ns3, disp_ns3)

    # Summary comparison table
    print("\n" + "=" * 60)
    print("SUMMARY: Generator Fingerprints")
    print("=" * 60)

    summary_path = os.path.join(args.out_dir, 'pde_fingerprint_comparison.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['PDE', 'S_norm', 'D_norm_mean', 'SD_ratio',
                     'R2_Im_vs_k', 'R2_Im_vs_k2', 'R2_Re_vs_k2',
                     'dominant_scaling'])

        # SWE
        dom = 'wave (Im~|k|)' if disp_swe['r2_im_vs_k'] > disp_swe['r2_re_vs_k2'] else 'dissipative (Re~|k|²)'
        w.writerow(['SWE', f'{swe["S_norm"]:.4f}', f'{swe["D_norm_mean"]:.4f}',
                     f'{swe["SD_ratio"]:.4f}',
                     f'{disp_swe["r2_im_vs_k"]:.4f}', f'{disp_swe["r2_im_vs_k2"]:.4f}',
                     f'{disp_swe["r2_re_vs_k2"]:.4f}', dom])
        print(f"  SWE:       S/D = {swe['SD_ratio']:.3f}  |  {dom}")

        # NS
        for key, (ns_info, ns_disp) in ns_results.items():
            label = f'NS_{key}'
            dom = 'wave (Im~|k|)' if ns_disp['r2_im_vs_k'] > ns_disp['r2_re_vs_k2'] else 'dissipative (Re~|k|²)'
            w.writerow([label, f'{ns_info["S_norm"]:.4f}', f'{ns_info["D_norm_mean"]:.4f}',
                         f'{ns_info["SD_ratio"]:.4f}',
                         f'{ns_disp["r2_im_vs_k"]:.4f}', f'{ns_disp["r2_im_vs_k2"]:.4f}',
                         f'{ns_disp["r2_re_vs_k2"]:.4f}', dom])
            print(f"  {label}: S/D = {ns_info['SD_ratio']:.3f}  |  {dom}")

    print(f"\nSaved to {summary_path}")
    print(f"\nKey prediction:")
    print(f"  SWE should have S/D >> NS (wave-dominated vs advection-dominated)")
    print(f"  SWE should have Im(λ)~|k| (linear dispersion)")
    print(f"  NS  should have Re(λ)~|k|² (viscous dissipation)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_swe', required=True)
    parser.add_argument('--ckpt_ns_1e5', default=None)
    parser.add_argument('--ckpt_ns_1e3', default=None)
    parser.add_argument('--out_dir', default='./swe_analysis')
    args = parser.parse_args()
    main(args)
