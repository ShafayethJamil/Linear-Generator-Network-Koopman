"""Generate the optical-fiber NLSE datasets by split-step Fourier integration, with the same pulses at every beta2.

Paper: Section 3.3, Appendix A.8.

Usage:
    python nlse/datagen.py --regime soliton --beta2_list -10 -15 -20 -25 -30 --save_dir data/nlse
        (add -5 for the below-resolution fiber of A.8)
Pulse-ensemble variants of A.8 (beta2 = -20; T0 in ps, P0 in W):
    NLSE_TAG=_wide   NLSE_T0_RANGE=5,10  python nlse/datagen.py --beta2 -20 --seed 42 --save_dir data/nlse_ensembles
    NLSE_TAG=_narrow NLSE_T0_RANGE=2,4   python nlse/datagen.py --beta2 -20 --seed 42 --save_dir data/nlse_ensembles
    NLSE_TAG=_hipow  NLSE_P0_RANGE=0.5,1 python nlse/datagen.py --beta2 -20 --seed 42 --save_dir data/nlse_ensembles
    NLSE_TAG=_seed43                     python nlse/datagen.py --beta2 -20 --seed 43 --save_dir data/nlse_ensembles
"""

import numpy as np
import os
import argparse
import csv
from scipy.io import savemat


# ============================================================
# Split-Step Fourier Solver (identical to nlse_datagen.py)
# ============================================================

def split_step_fourier(A0, z_span, dz, dt, beta2, gamma, alpha=0.0, beta3=0.0):
    N_t = len(A0)
    N_z = int(np.round(z_span / dz))
    omega = 2 * np.pi * np.fft.fftfreq(N_t, d=dt)
    D_linear = (-alpha / 2.0 + 1j * (beta2 / 2.0) * omega**2
                - 1j * (beta3 / 6.0) * omega**3)
    exp_half = np.exp(D_linear * dz / 2.0)
    total_save = 30
    save_every = max(1, N_z // total_save)
    A = A0.copy().astype(np.complex128)
    A_saved = [A.copy()]
    z_saved = [0.0]
    for step in range(N_z):
        A_f = np.fft.fft(A)
        A_f *= exp_half
        A = np.fft.ifft(A_f)
        intensity = np.abs(A)**2
        A *= np.exp(1j * gamma * intensity * dz)
        A_f = np.fft.fft(A)
        A_f *= exp_half
        A = np.fft.ifft(A_f)
        z_current = (step + 1) * dz
        if (step + 1) % save_every == 0:
            A_saved.append(A.copy())
            z_saved.append(z_current)
    if z_saved[-1] < z_span - dz / 2:
        A_saved.append(A.copy())
        z_saved.append(N_z * dz)
    return np.array(A_saved), np.array(z_saved)


# ============================================================
# Pulse generators (identical to nlse_datagen.py)
# ============================================================

def sech_pulse(t, T0, P0, C=0.0, t_offset=0.0):
    tau = (t - t_offset) / T0
    return np.sqrt(P0) / np.cosh(tau) * np.exp(-1j * C / 2.0 * tau**2)

def gaussian_pulse(t, T0, P0, C=0.0, t_offset=0.0, order=1):
    tau = (t - t_offset) / T0
    return np.sqrt(P0) * np.exp(-(1 + 1j * C) / 2.0 * tau**(2 * order))

def dual_pulse(t, T0, P0, separation, C=0.0):
    A1 = sech_pulse(t, T0, P0 / 2, C, t_offset=-separation / 2)
    A2 = sech_pulse(t, T0, P0 / 2, C, t_offset=+separation / 2)
    return A1 + A2


# ============================================================
# Dataset generation with configurable β₂
# ============================================================

def generate_sweep_dataset(beta2_value, regime, n_samples, seed=42):
    """
    Generate NLSE dataset at a specific β₂ value.

    CRITICAL: Uses deterministic seeding so that initial conditions
    are IDENTICAL across β₂ values. Only the propagation physics changes.
    """
    rng = np.random.RandomState(seed)

    # Fiber parameters (fixed except β₂)
    gamma = 1.3e-3        # 1/(W·m) — FIXED
    alpha_dB = 0.2e-3     # dB/m — FIXED
    alpha = alpha_dB * np.log(10) / 10.0
    beta3 = 0.1e-3        # ps³/m — FIXED

    # Time grid
    N_t = 256
    T_window = 100.0  # ps
    dt = T_window / N_t
    t = np.arange(N_t) * dt - T_window / 2.0

    # Regime parameters (same as original nlse_datagen.py)
    if regime == 'soliton':
        P0_range = (0.05, 1.0)
        T0_range = (2.0, 8.0)
        z_total = 5000.0
        dz = 2.0
    elif regime == 'nonlinear':
        P0_range = (0.5, 10.0)
        T0_range = (1.0, 5.0)
        z_total = 1000.0
        dz = 0.5
    else:
        raise ValueError(f"Regime must be 'soliton' or 'nonlinear', got '{regime}'")

    # optional pulse-ensemble ranges from the environment (unchanged if unset)
    if os.environ.get('NLSE_T0_RANGE'):
        T0_range = tuple(float(v) for v in os.environ['NLSE_T0_RANGE'].split(','))
    if os.environ.get('NLSE_P0_RANGE'):
        P0_range = tuple(float(v) for v in os.environ['NLSE_P0_RANGE'].split(','))
    print(f'  pulse ensemble: P0_range={P0_range} T0_range={T0_range}')
    # Fixed z_total for all β₂ — ensures identical dz_snap across sweep
    N_z_steps = int(z_total / dz)
    save_every = max(1, N_z_steps // 30)

    print(f"\n  β₂ = {beta2_value*1e3:.1f} ps²/km, regime={regime}")
    print(f"  z_total = {z_total/1000:.1f} km (fixed)")
    print(f"  z steps = {N_z_steps}, save_every = {save_every}, dz_snap = {save_every*dz:.0f} m")

    all_fields = []
    all_params = []

    for i in range(n_samples):
        P0 = rng.uniform(*P0_range)
        T0 = rng.uniform(*T0_range)
        C = rng.uniform(-2.0, 2.0)
        t_offset = rng.uniform(-3.0, 3.0)
        noise_amp = rng.uniform(0.001, 0.01)
        N_sol = np.sqrt(gamma * P0 * T0**2 / abs(beta2_value))

        shape_choice = rng.choice(['sech', 'gaussian', 'dual'], p=[0.5, 0.3, 0.2])
        if shape_choice == 'sech':
            A0 = sech_pulse(t, T0, P0, C, t_offset)
            shape = 'sech'
        elif shape_choice == 'gaussian':
            order = rng.choice([1, 2])
            A0 = gaussian_pulse(t, T0, P0, C, t_offset, order)
            shape = f'gauss_o{order}'
        else:
            sep = rng.uniform(5 * T0, 15 * T0)
            A0 = dual_pulse(t, T0, P0, sep, C)
            shape = 'dual'

        A0 += noise_amp * np.sqrt(P0) * (rng.randn(N_t) + 1j * rng.randn(N_t))

        A_z, z_out = split_step_fourier(A0, z_total, dz, dt,
                                         beta2_value, gamma, alpha, beta3)

        all_fields.append(A_z)
        all_params.append({
            'P0': P0, 'T0': T0, 'C': C, 'N_sol': N_sol,
            'shape': shape, 't_offset': t_offset, 'noise_amp': noise_amp,
        })

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n_samples} done")

    min_z = min(a.shape[0] for a in all_fields)
    fields = np.array([a[:min_z] for a in all_fields])
    z_out = z_out[:min_z]

    print(f"  Shape: {fields.shape}, z_snaps: {min_z}")

    return fields, z_out, t, all_params, {
        'beta2': beta2_value, 'gamma': gamma, 'alpha': alpha,
        'beta3': beta3, 'N_t': N_t, 'T_window': T_window,
        'dt': dt, 'z_total': z_total, 'dz': dz,
    }


def save_dataset(fields, z_out, t, params, fiber_params, tag, save_dir):
    """Save in Koopman-LGN compatible format."""
    N, N_z, N_t = fields.shape
    u_power = (np.abs(fields)**2).transpose(0, 2, 1).astype(np.float32)
    u_real = np.real(fields).transpose(0, 2, 1).astype(np.float32)
    u_imag = np.imag(fields).transpose(0, 2, 1).astype(np.float32)

    mat_path = os.path.join(save_dir, f'nlse_{tag}.mat')
    savemat(mat_path, {
        'u_power': u_power,
        'u_real': u_real,
        'u_imag': u_imag,
        't': t.astype(np.float32),
        'z': z_out.astype(np.float32),
        'fiber_beta2': fiber_params['beta2'],
        'fiber_gamma': fiber_params['gamma'],
        'fiber_alpha': fiber_params['alpha'],
        'fiber_beta3': fiber_params['beta3'],
    }, do_compression=True)
    print(f"  Saved: {mat_path}  ({u_power.shape})")

    meta_path = os.path.join(save_dir, f'nlse_{tag}_meta.csv')
    with open(meta_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample', 'P0', 'T0', 'C', 'N_sol', 'shape',
                     't_offset', 'noise_amp'])
        for i, p in enumerate(params):
            w.writerow([i, p['P0'], p['T0'], p['C'], p['N_sol'],
                         p['shape'], p['t_offset'], p['noise_amp']])

    return mat_path


def main():
    parser = argparse.ArgumentParser(description='β₂ Sweep Data Generator')
    parser.add_argument('--regime', default='soliton', choices=['soliton', 'nonlinear'])
    parser.add_argument('--beta2', type=float, default=None,
                        help='Single β₂ value in ps²/km (e.g., -10)')
    parser.add_argument('--beta2_list', nargs='+', type=float, default=None,
                        help='List of β₂ values in ps²/km')
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', default='data/nlse')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Default sweep values
    if args.beta2 is not None:
        beta2_list = [args.beta2]
    elif args.beta2_list is not None:
        beta2_list = args.beta2_list
    else:
        beta2_list = [-5, -10, -15, -20, -25, -30]

    # Convert from ps²/km to ps²/m
    beta2_values = [b * 1e-3 for b in beta2_list]

    print("=" * 70)
    print("β₂ SWEEP DATA GENERATION")
    print("=" * 70)
    print(f"Regime: {args.regime}")
    print(f"β₂ values: {beta2_list} ps²/km")
    print(f"Samples: {args.n_train + args.n_test} (train={args.n_train}, test={args.n_test})")
    print(f"Seed: {args.seed}")

    # Save sweep metadata
    sweep_meta_path = os.path.join(args.save_dir, f'sweep_meta_{args.regime}.csv')
    with open(sweep_meta_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['beta2_ps2_per_km', 'beta2_ps2_per_m', 'tag', 'mat_path',
                     'z_total_m', 'dz_snap_m', 'n_z_saved'])

        for beta2_val in beta2_values:
            beta2_km = beta2_val * 1e3  # back to ps²/km for display
            tag = f"b2_{abs(beta2_km):.0f}_{args.regime}" + os.environ.get("NLSE_TAG", "")

            n_total = args.n_train + args.n_test
            fields, z_out, t, params, fiber_params = generate_sweep_dataset(
                beta2_val, args.regime, n_total, seed=args.seed)

            mat_path = save_dataset(fields, z_out, t, params, fiber_params,
                                     tag, args.save_dir)

            # Compute dz_snap
            dz_snap = z_out[1] - z_out[0] if len(z_out) > 1 else 0
            w.writerow([beta2_km, beta2_val, tag, mat_path,
                         fiber_params['z_total'], dz_snap, len(z_out)])

    print(f"\n{'='*70}")
    print(f"Sweep metadata: {sweep_meta_path}")
    print(f"{'='*70}")
    print(f"\nNext: train models with, e.g.:")
    print(f"    python nlse/train_nlse.py --data {args.save_dir}/nlse_b2_20_soliton.mat \\")
    print(f"      --field complex --tag b2_20_s1 --seed 1 --save_dir checkpoints/nlse")


if __name__ == '__main__':
    main()
