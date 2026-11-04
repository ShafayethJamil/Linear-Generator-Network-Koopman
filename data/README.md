# Data

## Navier–Stokes (Li et al., Fourier Neural Operator benchmark)

Download the two files from the FNO authors' data folder, https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-, and place them in `data/ns/`. The ν = 10⁻³ file is distributed as `NavierStokes_V1e-3_N5000_T50.zip`, which contains `ns_V1e-3_N5000_T50.mat`.

| file | size | SHA-256 | used for |
|---|---|---|---|
| `NavierStokes_V1e-5_N1200_T20.mat` | 413 MB | `b4995d0180ec15f06878e17527fd3e08a8f504bbc059ee025caffad11a9a82c0` | ν = 10⁻⁵, T_out = 10 |
| `ns_V1e-3_N5000_T50.mat` | 3.85 GB | `78b8d9e83d767dc7050fb8145ee7e7f11e2d18d325bff9abc7f108ec3292ee78` | ν = 10⁻³, T_out = 40 |

Check your copy with `sha256sum data/ns/*.mat`.

## Optical fiber (nonlinear Schrödinger equation)

Generated, not downloaded. The generator uses a fixed seed (42), so the same pulses appear at every β₂:

```bash
python nlse/datagen.py --regime soliton --beta2_list -10 -15 -20 -25 -30 --save_dir data/nlse
```

The pulse-ensemble variants of Appendix A.8 (iii) are listed in the docstring of `nlse/datagen.py`.
