# dring Config Help

This file is the maintained reference for the YAML options used by `dring fit`.
Most users only need the required options. Advanced options can be omitted; the
code has conservative defaults.

## Command Line

Install fitting support before using the fit/analysis commands:

```bash
pip install "dring[fit]"
pip install "dring[fit,mpi]"  # for MPI runs
```

```bash
dring fit -c configs/HD163296_ring1.yaml
dring fit -c configs/HD163296_ring1.yaml --check-config
dring --help
```

`--check-config` loads the data, opacity, beams, fitting radii, model grid, and
parameter names without running UltraNest.

## Required Data Options

```yaml
data:
  bands:
    - wavelength_cm: 0.089
      band_name: band7
      angular_resolution_arcsec: 0.0667
      cal_error: 0.1
      profile_path: data/HD163296_band7.npz
    - wavelength_cm: 0.13
      band_name: band6
      angular_resolution_arcsec: 0.0428
      cal_error: 0.1
      profile_path: data/HD163296_band6.npz
```

- `bands`: one entry per wavelength band.
- `wavelength_cm`: observing wavelength for that band.
- `band_name`: human-readable label used in plot legends.
- `angular_resolution_arcsec`: circular/effective beam FWHM for that band.
- `cal_error`: fractional absolute flux calibration uncertainty.
- `profile_path`: named `.npz` profile for that band.

Each band profile `.npz` contains:

```text
radius      radius [au by default], shape (n_radius,)
intensity   intensity [Jy/sr], shape (n_radius,)
rms_error   rms uncertainty [Jy/sr], shape (n_radius,)
```

When using the Python API, these arrays can be supplied directly:

```python
config["data"]["bands"][0]["radius"] = radius
config["data"]["bands"][0]["intensity"] = intensity
config["data"]["bands"][0]["rms_error"] = rms_error
```

## Required Opacity Options

```yaml
opacity:
  size_opac: data/default_opacity_dsharp/size_opac.npy
  lam_opac: data/default_opacity_dsharp/lam_opac.npy
  k_abs_opac: data/default_opacity_dsharp/k_abs_opac.npy
  k_sca_opac: data/default_opacity_dsharp/k_sca_opac.npy
  g_sca_opac: data/default_opacity_dsharp/g_sca_opac.npy
  rhos_opac: data/default_opacity_dsharp/rhos_opac.npy
```

Opacity arrays use cgs units:

```text
size_opac    grain radius [cm]
lam_opac     wavelength [cm]
k_abs_opac   absorption opacity [cm^2/g]
k_sca_opac   scattering opacity [cm^2/g]
g_sca_opac   scattering asymmetry parameter
rhos_opac    internal grain density [g/cm^3]
```

## Required Observation Options

```yaml
distance_pc: 101.2
inclination_deg: 46.0
fit_radii_au: [63.0, 67.0, 71.0]
```

- `fit_radii_au`: shared fitting radii for all bands.

For band-dependent fitting radii, use:

```yaml
fit_radii_by_band_au:
  "0": [63.0, 67.0, 71.0]
  "1": [64.0, 67.0, 70.0]
  "2": [65.0, 67.0, 69.0]
```

Band indices are zero-based.

## Required Model Options

```yaml
model:
  stellar_mass_msun: 1.9
  ring_center_au: 67.0
  pressure_width_au: 14.4
  sigma_g: 21.0
  size_res: 100
  temperature_slope: -0.5
  size_distribution_q: -3.5
```

The radial model grid is automatic by default. The code uses the fitted radii,
`ring_center_au`, and the largest beam to choose a safe radial range. The number
of radial points is also automatic:

```text
dr <= min(beam_min / 5, pressure_width_au / 8, 1 au)
80 <= n_grid <= 300
```

Manual override:

```yaml
model:
  r_grid_au: [50.0, 80.0, 120]
```

Optional radial-grid tuning:

```yaml
model:
  r_grid_n: 120              # fixed number of grid points
  r_grid_beam_margin: 5.0    # radial padding in units of max beam
  r_grid_extra_margin_au: 0.0
  r_grid_beam_points: 5.0    # points per minimum beam FWHM
  r_grid_width_points: 8.0   # points per pressure width
  r_grid_max_dr_au: 1.0
  r_grid_min_dr_au: 0.05
  r_grid_min_n: 80
  r_grid_max_n: 300
```

## Priors

```yaml
priors:
  alpha: {type: flat, scale: log, min: 1.0e-6, max: 1.0e-1}
  vf: {type: flat, scale: log, min: 1.0, max: 1.0e4}
  eps: {type: flat, scale: log, min: 1.0e-3, max: 1.0}
  T: {type: flat, scale: linear, min: 5.0, max: 30.0}
```

Each prior supports:

```yaml
type: flat       # flat or gaussian
scale: linear   # linear or log
min: 5.0
max: 30.0
```

For a bounded Gaussian prior, add `center` and `sigma`:

```yaml
T:
  type: gaussian
  scale: linear
  center: 18.0
  sigma: 3.0
  min: 5.0
  max: 30.0
```

For log-scale Gaussian priors, `center` is in physical units. By default,
`sigma` is also interpreted in physical units and converted locally to dex. Set
`sigma_is_linear: false` to provide `sigma` directly in dex.

The sampled parameters are:

```text
alpha, vf, eps, T, c_lam1, c_lam2, ...
```

Calibration factors use a flat prior by default:

```text
c_lam = 1 +/- 3 * cal_error
```

The likelihood also includes the Gaussian calibration penalty unless disabled.

## Output And Sampler Options

```yaml
output_dir: results/HD163296_ring1
resume: overwrite
make_plots: true
min_num_live_points: 200
frac_remain: 0.01
```

Each run writes the resolved setup to:

```text
<output_dir>/config.yaml
```

This copied config is the recommended input for posterior analysis.

## Built-In Analysis Products

`dring fit` writes the usual UltraNest plots and, by default, also runs the
three packaged analysis products on rank 0:

```text
<output_dir>/corner.png
<output_dir>/derived_summary.txt
derived posterior summary printed to stdout
<output_dir>/intensity_bestfit.png
```

The packaged analysis commands are:

```bash
dring plot -r results/HD163296_ring1
dring derived -r results/HD163296_ring1
dring intensity -r results/HD163296_ring1
```

Optional YAML controls live under `analysis`. Omit them for the default style:

```yaml
analysis:
  corner:
    enabled: true
    output: corner.png
    columns: [0, 1, 2, 3]
    # Default false: diagonal panels show histograms only.
    show_diagonal_stats: false
    # Ranges are in sampled units; log-labeled axes are converted to log10.
    ranges:
      alpha: [1.0e-6, 1.0e-1]
      vf: [1.0, 1.0e4]
      eps: [1.0e-3, 1.0]
      T: [5.0, 30.0]
    # xlim/ylim use the same units as ranges. Log-labeled axes are converted.
    xlim:
      alpha: [1.0e-6, 1.0e-1]
    ylim:
      T: [5.0, 35.0]

  derived:
    enabled: true
    output: derived_summary.txt
    tau_wavelength_cm: 0.13
    # Automatic post-fit table evaluation uses 100 samples by default.
    # Set null to use every posterior sample.
    max_samples: 500

  intensity:
    enabled: true
    output: intensity_bestfit.png
    cmap: viridis
    cmap_range: [0.08, 0.92]
    xlim: [0.0, 200.0]
    ylim: [1.0e5, 1.0e11]
```

For the intensity figure, the default x-range is the full radial extent of the
input data. The default y-range is logarithmic, from one hundredth of the
minimum fitted positive intensity to ten times the maximum fitted positive
intensity. Band colors are sampled linearly from the configured colormap, so
the same plot works for any number of wavelengths.

For manual model debugging, every `DustRingModel` also has:

```python
model.run_model()
model.SDplot("sigma_dust.png")
```

This plots `log10(SigmaDust / dlnm)` on the model radial grid and grain-size
grid. In MPI runs, only rank 0 creates or writes this figure.

Nested sampler options can also be grouped:

```yaml
sampler:
  min_num_live_points: 400
  frac_remain: 0.01
  stepsampler: slice
  nsteps: 20
```

By default, RingFit uses UltraNest without an explicit step sampler, which is
the safer choice for clearly multi-modal posteriors. To enable slice sampling,
set:

```yaml
sampler:
  stepsampler: slice
```

## Advanced Defaults

These options can be omitted from normal configs:

```yaml
oversampling_correction: true
include_calibration_penalty: true
calibration_prior: flat
calibration_prior_nsigma: 3.0
reduced_chi2_dof: raw_data
sampler.stepsampler:
```

### Oversampling Correction

When `oversampling_correction: true`, the code inflates the rms at each fitted
radius:

```text
N_i = max(1, beam_eff_au / local_dr_i)
sigma_eff_i = sigma_rms_i * sqrt(N_i)
```

This preserves the likelihood form while reducing the overweighting of
oversampled radial-profile points.

## Continuum Scattering Formula

By default, the paper-release configuration uses the original Zhu et al.
2019-style slab-scattering continuum approximation:

```yaml
scattering_formula: zhu2019
```

The Kitade & Kataoka 2026 Stokes-I fitting formula is also available only as a
reference/comparison mode:

```yaml
scattering_formula: kataoka2026
```

For `kataoka2026`, the required `A_I`, `B_I`, `I_conv`, and `omega_I`
coefficient table is bundled locally with the package under
`ringfit/kataoka2026_coeffs/`, generated from the authors' official
`emergentintensity` Stokes-I RT tables and fitting formula. Runtime model
evaluation does not download data or refit coefficients.

This bundled table is provided for convenience, testing, and comparison. It is
not a substitute for the official Kitade & Kataoka data/code in rigorous
scientific fitting. Users who rely on `kataoka2026` should obtain the official
tables/code, validate or regenerate coefficients for their use case, and pass
their coefficient file explicitly:

```yaml
kataoka2026_coeffs: path/to/kataoka2026_stokesI_coeffs.npz
```

The opacity treatment still uses the effective-scattering approximation
`kappa_sca_eff = (1 - g) * kappa_sca`, which may be unreliable for strongly
forward-peaked scattering, especially porous or very large grains.
The official Stokes-I coefficient table is validated through `omega=0.9`; if
the effective albedo is between 0.9 and 1.0, coefficient interpolation is
clamped at the published upper grid edge rather than extrapolated.

## Optional Speed Dependency

If `numba` is installed, RingFit automatically JIT-compiles the inner
radiative-transfer loop used by `scattering_precomputed`. If `numba` is absent,
the code falls back to the pure NumPy implementation with no config change.

Install the optional dependency with:

```bash
pip install -r requirements-optional.txt
```

MPI note: each MPI rank may compile the function on its first likelihood call.
For long UltraNest runs this startup cost is usually negligible compared with
the full sampling time.
