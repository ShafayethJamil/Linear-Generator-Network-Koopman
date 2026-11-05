# LGN-KM: Linear Generator Network–Koopman

Code, checkpoints and a reproduction check for

> S. Jamil and R. Kapadia, *Linear Generator Networks for Nonlinear Partial Differential Equations*, NeurIPS 2026.

LGN-KM lifts a nonlinear field into a latent space and learns the continuous-time Koopman generator at every retained Fourier mode as L_k = S − D_k. S is skew-symmetric and shared across modes; D_k is a positive diagonal with wavenumber-dependent damping. Every eigenvalue therefore has Re(λ) ≤ 0, the spectrum can be read directly from the parameters, and any time t is one matrix exponential.

## Quick start: check the paper's numbers

```bash
pip install -r requirements.txt
bash scripts/download_checkpoints.sh       # all checkpoints of the paper (~300 MB) into checkpoints/
python verify.py                           # checkpoints and shipped result files only, CPU, a few seconds
python verify.py --data_dir data/ns        # also re-evaluates test errors (GPU recommended)
```

`--data_dir` is one folder. It must hold the two benchmark files. Optionally it can also hold the generated `rd_neumann_Du*.mat` (Table A.4 test errors) and the two `ns_*_fixed.mat` long-horizon files (Tables A.6/A.7); checks whose files are missing are skipped.

`verify.py` recomputes each number in the paper from the checkpoints or the shipped result files. It compares each value with the number as printed and reports PASS or FAIL. Checks are strict: a number that does not round to the printed value is reported as FAIL and never loosened.

`checkpoints/MANIFEST*.json` records, for every checkpoint, the original file it came from, its SHA-256, and how it was selected.

## Paper → code

| Paper | Code | Checkpoints (`checkpoints/…`) | Data |
|---|---|---|---|
| §2–3.1, Fig. 2, main Navier–Stokes models | `navier_stokes/train_lgnkm.py`, `navier_stokes/extract_SD.py` | `ns/lgnkm_nu{1e5,1e3}_s{0,1,2}.pt` | NS benchmark |
| §3.2, Fig. 3b–d, cross-viscosity statistics (3 seeds) | `verify.py` (`verify_ns`) | `ns/lgnkm_*` | — |
| §3.2, Fig. 3a, Table A.3, learnable exponent p | `navier_stokes/dissipation/lgn_flexD.py --Dparam freeexp` | `dissipation/best_koopman_flexD_freeexp_*` | NS benchmark |
| A.6, free diagonal and window-dependent exponent | `lgn_flexD.py --Dparam free`, `navier_stokes/dissipation/free_diagonal_fits.py` | `dissipation/best_koopman_flexD_free_*` | NS benchmark |
| §3.3, Fig. 4, A.8, Table A.5, optical fiber (NLSE) | `nlse/datagen.py`, `nlse/train_nlse.py`, `nlse/readout.py` | `nlse/b2_{10..30}_s{0,1,2}.pt`, `nlse/b2_5_s0.pt` | generated (`nlse/datagen.py`) |
| Table 1, FNO baseline | `navier_stokes/train_fno.py` | `ns/fno_nu{1e5,1e3}.pt` | NS benchmark |
| Table 1, DMD, LRAN and KNO baselines | `baselines/{dmd,lran,kno}.py` (outputs in `baselines/results/`) | `baselines/{lran,kno}_nu{1e3,1e5}_s{0,1,2}.pt` (DMD is deterministic) | NS benchmark |
| §4.1, Fig. 5a–b, per-step error and enstrophy over 200 steps | `long_horizon/fig5ab_rollout.py` | `ns/lgnkm_*_s0.pt`, `ns/fno_nu1e5.pt` | NS benchmark |
| §4.1, Fig. 5c, wall time vs horizon | `continuous_time/continuous_time.py` | `ns/lgnkm_nu1e5_s0.pt`, `ns/fno_nu1e5.pt` | NS benchmark |
| A.9, Table A.6, long-horizon accuracy | `long_horizon/run_pipeline.sh` (generate → energy gate → evaluate) | LGN-KM, FNO, `baselines/lran_nu1e5_s0.pt`, `baselines/kno_nu1e5_s0.pt` | regenerated, 200 trajectories |
| A.9, Table A.7, unseen time resolution | `long_horizon/run_pipeline.sh`, `long_horizon/eval_fine_dt.py` | `ns/lgnkm_nu1e5_s0.pt` | regenerated, 200 trajectories |
| §4.2, Fig. 6, A.3, Table A.1, cross-viscosity transfer | `transfer/transfer_experiment.py`, `transfer/fig6_metrics.py` | `transfer/best_{scratch,transfer_all,transfer_freezeS}_N*_s0.pt` | NS benchmark ν = 10⁻³ |
| A.2, Fig. A.1, channel dimension r | `navier_stokes/ablations/ablation_r.py`, `ablation_r2_fix.py` | `ns_ablations/r_sweep/` | NS benchmark |
| A.5, Table A.2, Fig. A.3, structural ablations | `navier_stokes/ablations/ablation_{unconstrained_L,S_only,D_only}.py` | `ns_ablations/structural/` | NS benchmark ν = 10⁻⁵ |
| A.10, Table A.8, sensitivity to r, M and normalization | `navier_stokes/ablations/analyze_spectral.py`, `train_normalized_input.py` | `ns_ablations/sensitivity/`, `ns_ablations/r_sweep/` | NS benchmark ν = 10⁻⁵ |
| A.4, Fig. A.2, FitzHugh–Nagumo, periodic | `fitzhugh_nagumo/{generate_fhn_periodic,train_fhn_periodic,analyze_fhn_spectrum,analyze_fhn_universality}.py` | `fhn/periodic_Du{0.01,0.001}_s0.pt` | generated |
| A.7, Table A.4, FitzHugh–Nagumo, no-flux (DCT-II lift) | `fitzhugh_nagumo/{generate_fhn_neumann,train_fhn_neumann}.py` | `fhn/neumann_Du{0.01,0.001}_s{0,1,2}.pt` | generated |

Each script's docstring gives the exact command that produced the paper's result.

The scripts are the ones that produced the paper. Hard-coded paths became arguments, and three known bugs are fixed and commented in the code:
- `continuous_time.py` compared step t with the wrong frame (an off-by-one); the timings it reports were not affected.
- The no-flux check in `generate_fhn_neumann.py` was a no-op; it now measures the wall-normal derivative.
- The long-horizon generators drew initial conditions at half the benchmark energy. They now use the Li et al. convention, and the pipeline refuses to evaluate unless the initial enstrophy is within 10% of the benchmark.

## Data

- **Navier–Stokes benchmark (Li et al.).** Download from the FNO authors' folder; see `data/README.md` for the link and SHA-256 checksums.
- **Generated datasets.** NLSE, FitzHugh–Nagumo and long-horizon data come from the generators, with fixed seeds:

```bash
python nlse/datagen.py --regime soliton --beta2_list -10 -15 -20 -25 -30 --save_dir data/nlse
python fitzhugh_nagumo/generate_fhn_periodic.py --Du 0.01 --out data/fhn/rd_Du0.01_N1200_T30.mat     # and --Du 0.001
python fitzhugh_nagumo/generate_fhn_neumann.py  --Du 0.01 --out data/fhn/rd_neumann_Du0.01_N1200_T30.mat   # and --Du 0.001
bash long_horizon/run_pipeline.sh          # ~2.2 h on 4 GPUs: 4 x 50 trajectories per file, energy gate, evaluation
```

## Protocol notes

- **Split.** The first 1000 trajectories train and the next 200 test, with no normalization (except the one A.10 row that is labelled as normalized).
- **No validation split.** Every reported test error is the minimum over training epochs. For converged runs this differs from the final epoch by less than 0.001. NLSE seeds 1–2 are final-epoch checkpoints recorded in a run ledger.
- **Transfer runs.** Fine-tuning uses 0.3× the learning rate of training from scratch, and the transfer results are single-seed.
- **Retraining.** A retrained model will not match the released checkpoints bit for bit (cuDNN is nondeterministic), but it should agree within the seed spread reported in the paper.

## Citation

See `CITATION.cff`.

## License

MIT, see `LICENSE`.
