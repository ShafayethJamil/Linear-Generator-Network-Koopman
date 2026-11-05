# LGN-KM: Linear Generator Networks for Nonlinear PDEs

Code and checkpoints for

> S. Jamil and R. Kapadia, *Linear Generator Networks for Nonlinear Partial Differential Equations*, NeurIPS 2026.

LGN-KM lifts a nonlinear field into a latent space and learns the continuous-time Koopman generator at every retained Fourier mode as L_k = S − D_k. S is skew-symmetric and shared across modes, and D_k is a positive diagonal with wavenumber-dependent damping. Every eigenvalue therefore has Re(λ) ≤ 0, so the spectrum can be read directly from the parameters, and any time t is evaluated with one matrix exponential.

## Quick start: check the paper's numbers

```bash
pip install -r requirements.txt
bash scripts/download_checkpoints.sh      # 24 checkpoints, 40 MB, into checkpoints/
python verify.py                          # 48 checks from the checkpoints alone, CPU, ~1 s
python verify.py --data_dir data/ns       # + test error on the benchmark (see data/README.md)
```

`verify.py` recomputes each headline number and compares it with the value printed in the paper. It covers:

- the Navier–Stokes eigenspectrum ranges, the maximum Re(λ), and the cross-viscosity statistics of S over three seeds;
- the optical-fiber dispersion readout for all 15 models;
- with data, the test errors in Table 1.

## What is in this release

| Paper | Code | Checkpoints |
|---|---|---|
| §2, §3.1–3.2: model and Navier–Stokes training | `navier_stokes/train_lgnkm.py` | `ns/lgnkm_nu{1e5,1e3}_s{0,1,2}.pt` |
| FNO baseline (A.1, Table 1) | `navier_stokes/train_fno.py` | `ns/fno_nu{1e5,1e3}.pt` |
| Eigenspectrum and S, d, α export (Fig. 2–3) | `navier_stokes/extract_SD.py` | the same |
| §3.3, Fig. 4, A.8: optical-fiber NLSE data, training, readout | `nlse/datagen.py`, `nlse/train_nlse.py`, `nlse/readout.py` | `nlse/b2_{10..30}_s{0,1,2}.pt`, `nlse/b2_5_s0.pt` |
| Headline numbers | `verify.py` | all of the above |

`checkpoints/MANIFEST.json` records, for every checkpoint, the original file it came from, its SHA-256, and how it was selected.

**Still to come.** These experiments will be added in the following weeks, from the same runs:

- the structural ablations (A.5) and the channel-dimension sweep (A.2);
- the learnable exponent and free diagonal (A.6);
- FitzHugh–Nagumo, periodic and no-flux (A.4, A.7);
- cross-viscosity transfer (§4.2, A.3);
- the DMD, LRAN and KNO baselines, the long-horizon tables and the fine-Δt queries (§4.1, A.9);
- the sensitivity table (A.10).

## Reproducing from scratch

Navier–Stokes, one model in about 40 min (ν = 10⁻⁵) or 97 min (ν = 10⁻³) on an RTX A5000:

```bash
python navier_stokes/train_lgnkm.py --data data/ns/NavierStokes_V1e-5_N1200_T20.mat --T_out 10 --seed 0 --tag nu1e5_s0
python navier_stokes/train_lgnkm.py --data data/ns/ns_V1e-3_N5000_T50.mat          --T_out 40 --seed 0 --tag nu1e3_s0
python navier_stokes/train_fno.py   --data data/ns/NavierStokes_V1e-5_N1200_T20.mat --T_out 10 --tag nu1e5_raw
python navier_stokes/extract_SD.py  --ckpt_1e5 checkpoints/best_koopman_nu1e5_s0.pt --ckpt_1e3 checkpoints/best_koopman_nu1e3_s0.pt
```

Optical fiber, about 5 min per model:

```bash
python nlse/datagen.py --regime soliton --beta2_list -10 -15 -20 -25 -30 --save_dir data/nlse
python nlse/train_nlse.py --data data/nlse/nlse_b2_20_soliton.mat --field complex --tag b2_20_s1 --seed 1 --save_dir checkpoints/nlse
python nlse/readout.py --ckpt checkpoints/nlse/final_koopman_nlse_b2_20_s1.pt --data data/nlse/nlse_b2_20_soliton.mat --beta2 -20
```

Retrained models will not match the released checkpoints bit for bit, because cuDNN is nondeterministic. They should match the paper's numbers to within the seed spread the paper reports.

## Protocol notes

- **Split and normalization.** The split is fixed: the first 1000 trajectories are used for training and the next 200 for testing. No normalization is applied.
- **Model selection.** No validation split is used. Every reported test error is the minimum over training epochs. For converged runs this differs from the final epoch by less than 0.0003.
- **NLSE checkpoints.** Seeds 1 and 2 are final-epoch checkpoints, recorded in a run ledger. Seed 0 predates the ledger and was selected on test loss; its best and final epochs agree within 10⁻⁴.
- **Readout constant.** The constant 13.56 is derived as 2 / (Ω₁₅² Δz), with Ω₁₅ = 2π·15/100 rad/ps and Δz = 0.166 km. It is not fitted.

## Citation

See `CITATION.cff`.

## License

MIT, see `LICENSE`.
