"""
Koopman-LGN for NLSE — 1D Fiber Optics Diagnostics
=====================================================
Adapted from Spatial Koopman-LGN (2D Navier-Stokes).

Changes from NS version:
    - SpectralConv2d → SpectralConv1d  (1D FFT over pulse time axis)
    - Encoder/Decoder: Conv2d → Conv1d
    - Grid: 2D (x,y) → 1D (t)
    - k2_norm: 2D modes → 1D modes
    - rfft2/irfft2 → rfft/irfft
    - Data loader for NLSE .mat files

UNCHANGED (the diagnostic core):
    - S = (P - Pᵀ)/2  (skew-symmetric, inter-frequency coupling / FWM)
    - D_k = diag(d₀ + α|ω|²/ω²_max)  (frequency-dependent loss)
    - L_k = S - D_k   (generator per frequency mode)
    - Propagation via expm(L_k · z)
    - All eigenvalue/cascade diagnostics identical

What the paper reads from this model (Section 3.3, Appendix A.8): the rotation
rates of S (its eigenfrequencies) and the routing of each optical frequency onto
them; |beta2| = 13.56 x (fastest rotation rate). See nlse/readout.py.

Paper models (complex field, final-epoch checkpoint; a run ledger is appended to
nlse/runs/ledger.jsonl):
    python nlse/train_nlse.py --data data/nlse/nlse_b2_20_soliton.mat \
        --field complex --tag b2_20_s1 --seed 1 --save_dir checkpoints/nlse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import matrix_exp
import sys

import json, hashlib, time, subprocess
def _sha(path, n=1<<24):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        h.update(f.read(n))  # first 16 MB + size: fast fingerprint
    h.update(str(os.path.getsize(path)).encode()); return h.hexdigest()[:16]
def _ledger(args, event, metrics=None):
    led=os.path.join(os.path.dirname(os.path.abspath(__file__)),'runs','ledger.jsonl')
    os.makedirs(os.path.dirname(led), exist_ok=True)
    row={'run_id':args.tag,'event':event,'time':time.strftime('%Y-%m-%dT%H:%M:%S'),
         'cmd':' '.join(sys.argv),'config':vars(args),'seed':args.seed}
    if event=='start':
        row.update({'script_sha':_sha(os.path.abspath(__file__)),
                    'data':{'path':args.data,'fp':_sha(args.data),'n_train':args.n_train,'n_test':args.n_test,'split':'contiguous [0:n_train] / [n_train:n_train+n_test]'},
                    'selection':'final-epoch checkpoint; test L2 logged every epoch but not used for selection','status':'running'})
    else:
        row.update({'metrics':metrics,'test_evals':args.epochs,'status':'ok'})
    with open(led,'a') as f: f.write(json.dumps(row)+'\n')
import numpy as np
import argparse, os, csv
from timeit import default_timer

# seeds are set in train() from --seed (ladder project edit)


# ============================================================
# Spectral Conv 1D  (was SpectralConv2d)
# ============================================================

class SpectralConv1d(nn.Module):
    """Fourier layer for 1D signals — spectral convolution in frequency domain."""
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_ch * out_ch)
        # Complex weights for low-frequency modes
        self.w = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x):
        # x: (B, C, N_t)
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x)  # (B, C, N//2+1)
        out_ft = torch.zeros(B, self.w.shape[1], N // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        # Multiply low-frequency modes
        out_ft[:, :, :self.modes] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :self.modes], self.w)
        return torch.fft.irfft(out_ft, n=N)


# ============================================================
# Encoder / Decoder  (1D versions)
# ============================================================

class Encoder1D(nn.Module):
    """Encode pulse snapshots → latent channels via 1D spectral convolutions."""
    def __init__(self, modes, width, r, Z_in):
        super().__init__()
        # Input: Z_in propagation snapshots + 1 grid coordinate
        self.fc0   = nn.Linear(Z_in + 1, width)
        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.w0    = nn.Conv1d(width, width, 1)
        self.w1    = nn.Conv1d(width, width, 1)
        self.proj  = nn.Conv1d(width, r, 1)

    def forward(self, x, grid):
        # x: (B, N_t, Z_in), grid: (B, N_t, 1)
        x = torch.cat([x, grid], dim=-1)       # (B, N_t, Z_in + 1)
        x = self.fc0(x).permute(0, 2, 1)       # (B, width, N_t)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        return self.proj(x)                     # (B, r, N_t)


class Decoder1D(nn.Module):
    """Decode latent channels → pulse power/field."""
    def __init__(self, r, out_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv1d(r, 128, 1), nn.GELU(), nn.Conv1d(128, out_channels, 1))

    def forward(self, z):
        out = self.net(z)  # (B, out_channels, N_t)
        if self.out_channels == 1:
            return out.squeeze(1)  # (B, N_t)
        return out  # (B, 2, N_t) for complex


# ============================================================
# Koopman-LGN 1D  (S-D core UNCHANGED from NS)
# ============================================================

class KoopmanLGN_1D(nn.Module):
    """
    Koopman-LGN for 1D dynamics (NLSE fiber optics).

    The diagnostic generator L_k = S - D_k is IDENTICAL to the NS version.
    Only the encoder/decoder and FFT dimension change.

    S: (r, r) skew-symmetric — encodes four-wave mixing coupling
    D_k: diag(d₀ + α|ω|²/ω²_max) — encodes frequency-dependent dissipation
    """
    def __init__(self, modes=16, width=32, r=32, Z_in=10, N_t=256, out_channels=1):
        super().__init__()
        self.modes, self.r, self.N_t = modes, r, N_t
        self.out_channels = out_channels
        self.encoder = Encoder1D(modes, width, r, Z_in)
        self.decoder = Decoder1D(r, out_channels=out_channels)

        # --- Diagnostic parameters (SAME as NS) ---
        self.P     = nn.Parameter(torch.randn(r, r) * 0.01)
        self.d     = nn.Parameter(torch.full((r,), -2.0))
        self.alpha = nn.Parameter(torch.full((r,), -3.0))

        # Frequency-mode normalization: |ω_k|² / ω²_max
        # For 1D rfft: modes 0, 1, ..., modes-1
        k = torch.arange(modes, dtype=torch.float32)
        k2_norm = k**2 / (modes**2)       # shape: (modes,)
        self.register_buffer('k2_norm', k2_norm)

    def get_generators(self):
        """
        Build L_k = S - D_k for each frequency mode k.

        Returns: (modes, r, r) tensor of generators
        """
        S  = (self.P - self.P.T) / 2.0              # (r, r)
        d0 = F.softplus(self.d)                       # (r,)
        a0 = F.softplus(self.alpha)                    # (r,)

        # D_k diagonal: d0 + alpha * |omega_k|^2 / omega_max^2
        # k2_norm: (modes,) → D_diag: (modes, r)
        D_diag = d0.unsqueeze(0) + a0.unsqueeze(0) * self.k2_norm.unsqueeze(-1)
        # D_diag: (modes, r)

        # Expand S to all modes: (modes, r, r)
        S_exp = S.unsqueeze(0).expand(self.modes, -1, -1).clone()
        D_mat = torch.diag_embed(D_diag)              # (modes, r, r)

        return S_exp - D_mat                           # (modes, r, r)

    def propagate(self, z0_ft, t, dt=1.0):
        """
        Propagate latent Fourier coefficients by distance t.

        z0_ft: (B, r, modes) — Fourier coefficients of latent state
        t: propagation distance (integer step or float)
        """
        L_all = self.get_generators()                  # (modes, r, r)
        eL = matrix_exp(L_all * t * dt)                # (modes, r, r)

        # z0_ft: (B, r, modes) → (B, modes, r, 1)
        z = z0_ft.permute(0, 2, 1).unsqueeze(-1)
        # eL: (modes, r, r) → (1, modes, r, r)
        z_t = (eL.unsqueeze(0).to(torch.cfloat) @ z).squeeze(-1)
        # z_t: (B, modes, r) → (B, r, modes)
        return z_t.permute(0, 2, 1)

    def forward(self, x, grid, Z_out, dt=1.0):
        """
        x:     (B, N_t, Z_in) — input pulse snapshots
        grid:  (B, N_t, 1) — normalized time coordinate
        Z_out: number of output propagation steps

        Returns: (B, Z_out, N_t) — predicted pulse power at each z
        """
        z0    = self.encoder(x, grid)                  # (B, r, N_t)
        z0_ft = torch.fft.rfft(z0)                     # (B, r, N_t//2+1)
        z0_ft_trunc = z0_ft[:, :, :self.modes]         # (B, r, modes)

        preds = []
        for t in range(1, Z_out + 1):
            zt_ft_trunc = self.propagate(z0_ft_trunc, t, dt)
            # Reconstruct full spectrum
            zt_ft = z0_ft.clone()
            zt_ft[:, :, :self.modes] = zt_ft_trunc
            z_t = torch.fft.irfft(zt_ft, n=self.N_t)  # (B, r, N_t)
            preds.append(self.decoder(z_t))             # (B, N_t)
        return torch.stack(preds, dim=1)                # (B, Z_out, N_t)

    def predict_at_z(self, x, grid, z_float, dt=1.0):
        """Continuous-distance evaluation — single expm at any real z."""
        z0    = self.encoder(x, grid)
        z0_ft = torch.fft.rfft(z0)
        z0_ft_trunc = z0_ft[:, :, :self.modes]
        zt_ft_trunc = self.propagate(z0_ft_trunc, z_float, dt)
        zt_ft = z0_ft.clone()
        zt_ft[:, :, :self.modes] = zt_ft_trunc
        z_t = torch.fft.irfft(zt_ft, n=self.N_t)
        return self.decoder(z_t)

    def get_max_re_eigenvalue(self):
        L_all = self.get_generators()
        eigs = torch.linalg.eigvals(L_all)
        return eigs.real.max().item()


# ============================================================
# Loss
# ============================================================

def paper_rel_l2_loss(pred, target):
    """Per-sample relative L2 over full spatiotemporal volume."""
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = target.reshape(B, -1)
    return ((p - t).norm(dim=1) / (t.norm(dim=1) + 1e-8)).mean()


# ============================================================
# Data loading for NLSE .mat files
# ============================================================

def load_nlse_data(mat_path, Z_in=10, Z_out=10, n_train=1000, n_test=200,
                   field='u_power'):
    """
    Load NLSE dataset.

    field options:
        'u_power' — |A|² power (1 channel per snapshot)
        'complex' — [Re(A), Im(A)] interleaved (2 channels per snapshot)
                     Z_in snapshots → 2*Z_in input channels
    """
    print(f"Loading {mat_path} ...")
    try:
        import h5py
        with h5py.File(mat_path, 'r') as f:
            if field == 'complex':
                u_re = np.array(f['u_real'], dtype=np.float32)
                u_im = np.array(f['u_imag'], dtype=np.float32)
            else:
                u = np.array(f[field], dtype=np.float32)
            t_axis = np.array(f['t'], dtype=np.float32).flatten()
            z_axis = np.array(f['z'], dtype=np.float32).flatten()
    except Exception:
        import scipy.io
        data = scipy.io.loadmat(mat_path)
        if field == 'complex':
            u_re = data['u_real'].astype(np.float32)
            u_im = data['u_imag'].astype(np.float32)
        else:
            u = data[field].astype(np.float32)
        t_axis = data['t'].flatten().astype(np.float32)
        z_axis = data['z'].flatten().astype(np.float32)

    if field == 'complex':
        # u_re, u_im: (N, N_t, N_z)
        # Interleave: for each z-snapshot, stack [Re, Im] → 2 channels
        # Result: (N, N_t, 2*N_z) with [..., 0]=Re(z0), [..., 1]=Im(z0), [..., 2]=Re(z1), ...
        N, N_t, N_z = u_re.shape
        u = np.zeros((N, N_t, 2 * N_z), dtype=np.float32)
        u[:, :, 0::2] = u_re
        u[:, :, 1::2] = u_im
        print(f"  Field: 'complex' [Re(A), Im(A)], shape: {u.shape}")
        print(f"  → {N_z} snapshots × 2 channels = {2*N_z} input channels")
        # Effective Z_in/Z_out doubled
        Z_in_eff = 2 * Z_in
        Z_out_eff = 2 * Z_out
        out_channels = 2
    else:
        N, N_t, N_z = u.shape
        print(f"  Field: '{field}', shape: {u.shape}")
        Z_in_eff = Z_in
        Z_out_eff = Z_out
        out_channels = 1

    print(f"  N={N}, N_t={N_t}, N_z={N_z}")
    print(f"  t range: [{t_axis[0]:.1f}, {t_axis[-1]:.1f}] ps")
    print(f"  z range: [{z_axis[0]:.0f}, {z_axis[-1]:.0f}] m")

    assert N_z >= Z_in + Z_out, \
        f"Not enough z snapshots: N_z={N_z}, need Z_in+Z_out={Z_in+Z_out}"

    u = torch.from_numpy(u)
    grid_1d = torch.linspace(0, 1, N_t).unsqueeze(-1)

    x_train = u[:n_train,               :, :Z_in_eff]
    y_train = u[:n_train,               :, Z_in_eff:Z_in_eff+Z_out_eff]
    x_test  = u[n_train:n_train+n_test, :, :Z_in_eff]
    y_test  = u[n_train:n_train+n_test, :, Z_in_eff:Z_in_eff+Z_out_eff]

    print(f"  Train: {x_train.shape} → {y_train.shape}")
    print(f"  Test:  {x_test.shape} → {y_test.shape}")

    return x_train, y_train, x_test, y_test, grid_1d, N_t, out_channels


# ============================================================
# Training
# ============================================================

def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    _ledger(args, 'start')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    x_train, y_train, x_test, y_test, grid, N_t, out_channels = load_nlse_data(
        args.data, args.Z_in, args.Z_out, args.n_train, args.n_test,
        field=args.field)

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test,  y_test  = x_test.to(device),  y_test.to(device)
    grid = grid.to(device)
    n_train, n_test = x_train.shape[0], x_test.shape[0]

    # Effective Z_in: 2*Z_in for complex field, Z_in for power
    Z_in_eff = x_train.shape[-1]

    model = KoopmanLGN_1D(
        modes=args.modes, width=args.width, r=args.r,
        Z_in=Z_in_eff, N_t=N_t, out_channels=out_channels).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Koopman-LGN 1D params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, f'curve_koopman_nlse_{args.tag}.csv')
    with open(csv_path, 'w', newline='') as cf:
        csv.writer(cf).writerow(['epoch', 'train_l2', 'test_l2',
                                  'max_re_eig', 'stable', 'time_s'])

    best_test_l2 = float('inf')
    print(f"\n[NLSE {args.tag}] Z_in={args.Z_in} (eff={Z_in_eff}) Z_out={args.Z_out} "
          f"r={args.r} modes={args.modes} epochs={args.epochs}")
    print(f"  field={args.field} out_channels={out_channels}\n")

    for ep in range(args.epochs):
        model.train()
        t0 = default_timer()
        train_l2 = 0.0

        perm = torch.randperm(n_train)
        for i in range(0, n_train, args.batch):
            idx = perm[i:i+args.batch]
            xb = x_train[idx]
            yb = y_train[idx]
            Bb = xb.shape[0]
            g  = grid.unsqueeze(0).expand(Bb, -1, -1)

            pred = model(xb, g, args.Z_out)

            # Reshape target to match prediction
            if out_channels == 1:
                # pred: (B, Z_out, N_t), target: (B, N_t, Z_out) → (B, Z_out, N_t)
                target = yb.permute(0, 2, 1)
            else:
                # pred: (B, Z_out, 2, N_t) from stacking decoder (B, 2, N_t)
                # target: (B, N_t, 2*Z_out) → reshape to (B, N_t, Z_out, 2) → (B, Z_out, 2, N_t)
                target = yb.reshape(Bb, N_t, args.Z_out, 2).permute(0, 2, 3, 1)

            loss = paper_rel_l2_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_l2 += loss.item() * Bb

        scheduler.step()
        train_l2 /= n_train

        # Eval
        model.eval()
        with torch.no_grad():
            test_l2 = 0.0
            for i in range(0, n_test, args.batch):
                xb = x_test[i:i+args.batch]
                yb = y_test[i:i+args.batch]
                Bb = xb.shape[0]
                g  = grid.unsqueeze(0).expand(Bb, -1, -1)
                pred = model(xb, g, args.Z_out)
                if out_channels == 1:
                    target = yb.permute(0, 2, 1)
                else:
                    target = yb.reshape(Bb, N_t, args.Z_out, 2).permute(0, 2, 3, 1)
                test_l2 += paper_rel_l2_loss(pred, target).item() * Bb
            test_l2 /= n_test
            max_re = model.get_max_re_eigenvalue()
            stable = 1 if max_re <= 0 else 0

        t1 = default_timer()

        with open(csv_path, 'a', newline='') as cf:
            csv.writer(cf).writerow([ep+1, f'{train_l2:.6f}', f'{test_l2:.6f}',
                                     f'{max_re:.6f}', stable, f'{t1-t0:.2f}'])

        if (ep + 1) % 10 == 0:
            flag = " <-- BEST" if test_l2 < best_test_l2 else ""
            stab = "stable" if stable else "UNSTABLE!"
            print(f"[{ep+1:04d}] train={train_l2:.4f} test={test_l2:.4f} "
                  f"Re(λ)_max={max_re:.5f} {stab}{flag}")

        if test_l2 < best_test_l2:
            best_test_l2 = test_l2
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f'best_koopman_nlse_{args.tag}.pt'))

    torch.save(model.state_dict(), os.path.join(args.save_dir, f'final_koopman_nlse_{args.tag}.pt'))
    _ledger(args, 'end', metrics={'test_l2_final': float(test_l2), 'test_l2_min_over_epochs': float(best_test_l2), 'train_l2_final': float(train_l2)})
    print(f"\n{'='*60}")
    print(f"DONE [Koopman NLSE {args.tag}]  best test_l2 = {best_test_l2:.4f}  final test_l2 = {test_l2:.4f}")
    print(f"{'='*60}")


# ============================================================
# SD Extraction (for diagnostics after training)
# ============================================================

def extract_diagnostics(args):
    """Extract S, D, eigenspectrum from trained checkpoint."""
    device = torch.device('cpu')

    state = torch.load(
        os.path.join(args.save_dir, f'final_koopman_nlse_{args.tag}.pt'),
        map_location=device, weights_only=False)

    P = state['P']
    d_raw = state['d']
    alpha_raw = state['alpha']

    r = P.shape[0]
    S = (P - P.T) / 2.0
    d0 = F.softplus(d_raw)
    a0 = F.softplus(alpha_raw)

    # Frequency modes
    modes = state['k2_norm'].shape[0]
    k2_norm = state['k2_norm']

    print(f"\n{'='*60}")
    print(f"NLSE DIAGNOSTICS — {args.tag}")
    print(f"{'='*60}")
    print(f"  r = {r}, modes = {modes}")
    print(f"  S Frobenius norm: {S.norm():.4f}")
    print(f"  d₀ (softplus): mean={d0.mean():.4f}, range=[{d0.min():.4f}, {d0.max():.4f}]")
    print(f"  α (softplus):  mean={a0.mean():.4f}, range=[{a0.min():.4f}, {a0.max():.4f}]")

    # Eigenspectrum for all modes
    all_k2 = []
    all_re = []
    all_im = []

    for ki in range(modes):
        D_diag = d0 + a0 * k2_norm[ki]
        L = S - torch.diag(D_diag)
        eigs = torch.linalg.eigvals(L)
        order = torch.argsort(eigs.real, descending=True)
        eigs_sorted = eigs[order]

        all_k2.append(ki**2)  # |ω_k|² proxy
        all_re.append(eigs_sorted.real.numpy())
        all_im.append(eigs_sorted.imag.numpy())

    all_k2 = np.array(all_k2)
    all_re = np.array(all_re)
    all_im = np.array(all_im)

    # Dominant eigenvalue regression: Re(λ) vs ω²
    dom_re = all_re[:, 0]
    mask = all_k2 > 0
    if mask.sum() > 2:
        coeffs = np.polyfit(all_k2[mask], dom_re[mask], 1)
        slope, intercept = coeffs
        ss_res = np.sum((dom_re[mask] - (slope * all_k2[mask] + intercept))**2)
        ss_tot = np.sum((dom_re[mask] - dom_re[mask].mean())**2)
        r2 = 1 - ss_res / ss_tot
    else:
        slope, intercept, r2 = 0, 0, 0

    n_eigs = all_re.size
    pct_stable = (all_re.flatten() <= 0).sum() / n_eigs * 100
    max_re = all_re.max()

    print(f"\n  DISPERSION RELATION (analog of Kolmogorov):")
    print(f"    R² (Re(λ) vs ω²):  {r2:.4f}")
    print(f"    Slope:              {slope:.6f}")
    print(f"    Total eigenvalues:  {n_eigs}")
    print(f"    Stable (Re≤0):      {pct_stable:.1f}%")
    print(f"    Max Re(λ):          {max_re:.6f}")

    # Save CSVs
    save_dir = args.save_dir

    # Eigenspectrum
    csv_path = os.path.join(save_dir, f'eigenspectrum_nlse_{args.tag}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['k', 'k_sq', 'k2_norm_val', 'dom_re', 'dom_im',
                     'all_re_min', 'all_re_max', 'D_diag_mean'])
        for ki in range(modes):
            D_diag_vals = d0 + a0 * k2_norm[ki]
            w.writerow([ki, ki**2, k2_norm[ki].item(),
                         all_re[ki, 0], all_im[ki, 0],
                         all_re[ki].min(), all_re[ki].max(),
                         D_diag_vals.mean().item()])
    print(f"  Saved: {csv_path}")

    # S matrix
    csv_path = os.path.join(save_dir, f'S_nlse_{args.tag}.csv')
    np.savetxt(csv_path, S.numpy(), delimiter=',')
    print(f"  Saved: {csv_path}")

    # Channel parameters
    csv_path = os.path.join(save_dir, f'channels_nlse_{args.tag}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['channel', 'd_raw', 'd0_softplus', 'alpha_raw', 'alpha_softplus'])
        for i in range(r):
            w.writerow([i, d_raw[i].item(), d0[i].item(),
                         alpha_raw[i].item(), a0[i].item()])
    print(f"  Saved: {csv_path}")

    # Summary
    csv_path = os.path.join(save_dir, f'diagnostics_nlse_{args.tag}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['R2_dispersion', r2])
        w.writerow(['slope', slope])
        w.writerow(['intercept', intercept])
        w.writerow(['n_eigenvalues', n_eigs])
        w.writerow(['pct_stable', pct_stable])
        w.writerow(['max_re_eigenvalue', max_re])
        w.writerow(['S_frobenius', S.norm().item()])
        w.writerow(['d0_mean', d0.mean().item()])
        w.writerow(['alpha_mean', a0.mean().item()])
    print(f"  Saved: {csv_path}")

    return r2, slope


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     required=True, help='Path to nlse_*.mat')
    parser.add_argument('--tag',      default='soliton')
    parser.add_argument('--field',    default='u_power',
                        help='Which field to train on: u_power, u_real, u_spec_log')
    parser.add_argument('--Z_in',     type=int,   default=10,
                        help='Number of input propagation snapshots')
    parser.add_argument('--Z_out',    type=int,   default=10,
                        help='Number of output propagation steps to predict')
    parser.add_argument('--n_train',  type=int,   default=1000)
    parser.add_argument('--n_test',   type=int,   default=200)
    parser.add_argument('--r',        type=int,   default=32)
    parser.add_argument('--modes',    type=int,   default=16,
                        help='Number of Fourier modes (frequency components)')
    parser.add_argument('--width',    type=int,   default=32)
    parser.add_argument('--epochs',   type=int,   default=500)
    parser.add_argument('--batch',    type=int,   default=20)
    parser.add_argument('--lr',       type=float, default=1e-3)
    parser.add_argument('--save_dir', default='./checkpoints_nlse')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--extract_only', action='store_true',
                        help='Skip training, just extract diagnostics from checkpoint')
    args = parser.parse_args()

    if args.extract_only:
        extract_diagnostics(args)
    else:
        train(args)
        print("\nExtracting diagnostics from FINAL-epoch checkpoint...")
        extract_diagnostics(args)
