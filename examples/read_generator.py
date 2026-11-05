"""Read the physics out of a trained LGN-KM model in a few lines.

The generator L_k = S - D_k is stored directly in the checkpoint: P (S = (P - P^T)/2),
d and alpha (D_k = diag(softplus(d) + softplus(alpha) |k|^2 / k_max^2)). No data needed.

Usage:
    python examples/read_generator.py
"""
import numpy as np
import torch
import torch.nn.functional as F


def generator(sd):
    """All per-mode generators L_k, shape (M, M, r, r), from a checkpoint state dict."""
    S = (sd['P'] - sd['P'].T) / 2
    D = F.softplus(sd['d']) + F.softplus(sd['alpha']) * sd['k2_norm'][..., None]
    return S - torch.diag_embed(D)


# Navier-Stokes turbulence: the eigenvalues of L_k are decay rates (Re) and frequencies (Im).
for nu in ['1e5', '1e3']:
    sd = torch.load(f'checkpoints/ns/lgnkm_nu{nu}_s0.pt', map_location='cpu')
    lam = torch.linalg.eigvals(generator(sd).double())
    print(f"Navier-Stokes nu=1e-{nu[-1]}: {lam.numel()} eigenvalues, "
          f"Re in [{lam.real.min():.2f}, {lam.real.max():.4f}] (all stable), "
          f"Im in +/-{lam.imag.abs().max():.2f}")

# Optical fiber: the fastest rotation rate of S gives the group-velocity dispersion,
# |beta2| = 2 f_max / (Omega_15^2 dz), with nothing fitted.
omega15, dz = 2 * np.pi * 15 / 100, 0.166          # band edge (rad/ps), snapshot spacing (km)
for b in [10, 20, 30]:
    sd = torch.load(f'checkpoints/nlse/b2_{b}_s1.pt', map_location='cpu')
    S = ((sd['P'] - sd['P'].T) / 2).double()
    f_max = torch.linalg.eigvals(S).imag.abs().max().item()
    print(f"Fiber with |beta2| = {b} ps^2/km: read from the generator {2 * f_max / (omega15**2 * dz):.1f} ps^2/km")
