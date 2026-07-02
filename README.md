# dring

`dring` is a fast semi-analytic physical model for dust-trapping rings in
protoplanetary disks. It is designed to connect multi-wavelength ring
continuum observations to physical parameters such as turbulence and fragmentation
velocity.

The package can be used at three levels: build a dust-ring model, forward-model
multi-wavelength intensity profiles, or run Bayesian fitting.

## Installation

For users who only want the physical dust-ring model and forward profiles:

```bash
pip install dring
```

For fitting observed radial profiles with the Bayesian module:

```bash
pip install "dring[fit]"
```

For MPI acceleration on clusters:

```bash
pip install "dring[fit,mpi]"
```

For local development from this repository, use the same extras but replace
`dring` with `-e .`:

```bash
pip install -e .
pip install -e ".[fit]"
pip install -e ".[fit,mpi]"
```

The base install depends only on NumPy. The fitting extra adds SciPy, PyYAML,
UltraNest, HDF5 support, and Matplotlib. MPI is optional and only installed
with the `mpi` extra.

## Quick Start

Before fitting, check the YAML config reference:

```bash
dring config-help
```

The full reference is maintained in `docs/config_help.md`. It explains the
input data format, model parameters, priors, units, and analysis options.

Build a dust-trapping ring model directly from physical parameters:

```python
import dring

model = dring.make_model(
    alpha=1e-3,
    vf=100.0,
    sigma_g=10.5,
    eps=0.03,
    pressure_width_au=14.4,
    temperature=20.0,
    ring_center_au=67.0,
    stellar_mass_msun=1.9,
)

print(model.SigmaDust.shape)
model.SDplot("quickstart_sigma_dust.png")
```

To run a small fitting example from this repository:

```bash
python -m dring fit -c configs/HD163296_ring1_eg.yaml --check-config
python -m dring fit -c configs/HD163296_ring1_eg.yaml
```

The fit writes posterior plots, derived quantities, and the best-fit intensity
comparison to the configured result directory.

The configuration files and input data used for the paper are included
under `configs/` and `data/`.

## Layer 1: Model

The model layer reconstructs the dust surface-density distribution from manual
physical parameters. It does not require opacity tables, observed data,
UltraNest, or MPI.

```python
import dring

model = dring.make_model(
    alpha=1e-3,
    vf=100.0,
    sigma_g=10.5,
    eps=0.03,
    pressure_width_au=14.4,
    temperature=20.0,
    ring_center_au=67.0,
    stellar_mass_msun=1.9,
)

r_grid_cm = model.r_grid
size_grid_cm = model.size
sigma_dust = model.SigmaDust
temperature_profile = model.T
```

Useful model products:

```text
model.r_grid      radial grid [cm]
model.size        grain-size grid [cm]
model.SigmaDust   dust surface-density distribution, shape (n_r, n_size)
model.T           temperature profile [K]
```

To inspect the generated dust distribution:

```python
model.SDplot("sigma_dust.png")
```

`SDplot` uses Matplotlib, so install the plotting/fitting extra if your base
environment does not already provide it.

For a quick visual exploration of the model parameters, try the interactive
model explorer: **https://dring-model-interactive.streamlit.app/**

## Layer 2: Radiative Transfer And Beaming

The forward-profile layer turns a dust model into intensity profiles. It
exposes the intermediate model products, so users can connect `dring` to their
own radiative-transfer, MCMC, or plotting tools.

The paper-release/default continuum treatment is the original Zhu et al.
2019-style slab-scattering approximation:

```yaml
scattering_formula: zhu2019
```

An optional Kitade & Kataoka 2026 mode (`kataoka2026`) is included only as a
reference and comparison tool. For convenience, `dring` bundles a small
coefficient table generated from the public Kitade & Kataoka 2026
`emergentintensity` Stokes-I tables, so users can try the formula without
downloading external files:

```yaml
scattering_formula: kataoka2026
```

This bundled table is not intended to replace the authors' official data/code
for rigorous work. If a project relies scientifically on the Kitade & Kataoka
2026 formula, users should consult the official `emergentintensity` release,
build or validate the coefficient table for their use case, and pass it
explicitly with `kataoka2026_coeffs`.

```python
r_au, raw_intensity = dring.compute_profile(
    model=model,
    wavelengths_cm=[0.089, 0.13, 0.32],
    opacity={
        "size_opac": size_opac,
        "lam_opac": lam_opac,
        "k_abs_opac": k_abs_opac,
        "k_sca_opac": k_sca_opac,
        "g_sca_opac": g_sca_opac,
        "rhos_opac": rhos_opac,
    },
    inclination_deg=46.0,
)
```

This returns the model radial grid in au and the raw, unconvolved intensity
profiles. Beam convolution is available through the lower-level helper:

```python
from ringfit.model import convolve_radial_profile

convolved = convolve_radial_profile(r_au, raw_intensity[0], beam_fwhm_au=8.0)
```

Opacity inputs are split arrays in cgs units:

```text
size_opac    grain radius [cm], shape (n_size,)
lam_opac     wavelength [cm], shape (n_lam,)
k_abs_opac   absorption opacity [cm^2/g], shape (n_size, n_lam)
k_sca_opac   scattering opacity [cm^2/g], shape (n_size, n_lam)
g_sca_opac   scattering asymmetry parameter, shape (n_size, n_lam)
rhos_opac    internal density [g/cm^3], scalar
```

The code does not auto-convert microns to cm. Convert before saving arrays;
this keeps the public interface boring and hard to misuse.

## Layer 3: Bayesian Module

Install the fitting extra to use YAML configs, UltraNest sampling, result
directories, and built-in analysis products:

```bash
pip install "dring[fit]"
dring demo
dring fit -c configs/HD163296_ring1.yaml --check-config
dring fit -c configs/HD163296_ring1.yaml
```

The same CLI is available as a module:

```bash
python -m dring fit -c configs/HD163296_ring1.yaml
```

The fitting extra requires UltraNest (`ultranest>=3.6`). Recent UltraNest
versions support NumPy 2; if you intentionally use an older UltraNest 3.x stack
on an HPC system, keep NumPy below 2 in that environment.

Each fit writes the resolved YAML file to the result directory as
`config.yaml`. Keep this file with the chains; the analysis helpers can then
read the posterior and the exact fitting setup from one folder.

By default, `dring fit` writes:

```text
<output_dir>/config.yaml
<output_dir>/corner.png
<output_dir>/derived_summary.txt
<output_dir>/intensity_bestfit.png
```

The installed analysis commands are:

```bash
dring plot -r results/HD163296_ring1
dring derived -r results/HD163296_ring1
dring intensity -r results/HD163296_ring1
```

Plot limits and analysis options can be set under `analysis.corner`,
`analysis.derived`, and `analysis.intensity` in the saved YAML. See
`docs/config_help.md` for the full config reference.

## MPI On Clusters

MPI is optional but recommended for long Bayesian fits. Install fitting and MPI
support together:

```bash
pip install "dring[fit,mpi]"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mpiexec -n 8 dring fit -c configs/HD163296_ring1.yaml
```

On older HPC systems, install in stages inside a disposable virtual
environment. This keeps the environment clean and makes compiler errors easier
to diagnose:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

# Pure/easy fitting dependencies first.
python -m pip install "numpy>=1.20" "scipy>=1.7" pyyaml "h5py>=3.7" matplotlib

# UltraNest may build from source on HPC; old GCC needs C99 enabled.
export CFLAGS="-std=c99"
python -m pip install "ultranest>=3.6"

# MPI is required for cluster parallel runs.
python -m pip install "mpi4py>=3"

# Install dring itself without asking pip to revisit dependencies.
python -m pip install -e . --no-deps
```

`CFLAGS` is only needed while installing UltraNest; it is not required at run
time. If you see a NumPy binary-compatibility error after changing NumPy
versions, reinstall NumPy and UltraNest in the same environment. If `mpi4py`
fails, check that the MPI compiler wrapper (`mpicc`) is available in the active
environment.

If a cluster already provides a working scientific Python/MPI stack, create a
disposable venv with `--system-site-packages` and install only `dring`:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -c "import numpy, scipy, ultranest, mpi4py; print(ultranest.__version__)"
pip install -e . --no-deps
```

Deleting `.venv` removes the `dring` install and leaves the site environment
untouched.

## Input Profile

Observation inputs are grouped by band. Each band owns its wavelength, angular
resolution, calibration uncertainty, radial grid, intensity, and rms
uncertainty:

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

Each `profile_path` is a named `.npz` containing one band:

```text
radius      radius [au by default], shape (n_radius,)
intensity   intensity [Jy/sr], shape (n_radius,)
rms_error   rms uncertainty [Jy/sr], shape (n_radius,)
```

For Python API use, the arrays can be supplied directly in each band dict
instead of through `profile_path`.

## Main Config Knobs

Edit one of the YAML files in `configs/`:

- `data.bands`: one entry per wavelength band.
- `opacity`: split opacity arrays in cgs units.
- `fit_radii_au`: shared fitting radii for all bands.
- `fit_radii_by_band_au`: optional band-dependent fitting radii.
- `model`: stellar mass, pressure-bump center, width, radial grid, and related model settings.
- `priors`: fitting ranges for `alpha`, `vf`, `eps`, and `T`.

The `model` block stores fixed source/model settings. The sampled physical
parameters are controlled by the `priors` block.

Common units used in the YAML files are:

```text
wavelength_cm                 observing wavelength [cm]
angular_resolution_arcsec      beam FWHM or effective 1D resolution [arcsec]
distance_pc                    source distance [pc]
inclination_deg                disk inclination [deg]
fit_radii_au                   radii used in the likelihood [au]
model.stellar_mass_msun        stellar mass [Msun]
model.ring_center_au           pressure-bump center [au]
model.pressure_width_au        Gaussian pressure width [au]
model.sigma_g                  gas surface density at the ring center [g/cm^2]
model.a_min, model.a_max       grain-size grid limits [cm]
priors.alpha                   turbulence parameter, dimensionless
priors.vf                      fragmentation velocity [cm/s]
priors.eps                     dust-to-gas ratio at the ring center, dimensionless
priors.T                       temperature normalization [K]
```

Advanced options can be omitted. By default, `oversampling_correction` is true,
`include_calibration_penalty` is true, and the calibration factors use a flat
`1 +/- 3*cal_error` sampling prior with a Gaussian calibration penalty in the
likelihood.

The sampled parameters are:

```text
alpha, vf, eps, T, c_lam1, c_lam2, ...
```

where each `c_lam*` is the band-dependent multiplicative calibration factor.

## Beam Smoothing And Oversampling

For model smoothing, the code uses the effective 1D angular resolution supplied
for each band:

```text
beam_eff = angular_resolution_arcsec
```

When `oversampling_correction: true`, the likelihood uses method A: it inflates
the rms uncertainty at each fitted radius rather than directly rescaling
chi-square:

```text
local_dr_i = local spacing of the fitted radii, allowing non-uniform sampling
N_i = max(1, beam_eff_au / local_dr_i)
sigma_eff_i = sigma_rms_i * sqrt(N_i)
```

For a single fitted radius in a band, `N_i = 1`.

## Prior Format

Priors are explicit dictionaries. The default examples use flat priors; bounded
Gaussian priors are also supported:

```yaml
priors:
  alpha: {type: flat, scale: log, min: 1.0e-6, max: 1.0e-1}
  vf: {type: flat, scale: log, min: 1.0, max: 1.0e4}
  eps: {type: flat, scale: log, min: 1.0e-3, max: 1.0}
  T:
    type: gaussian
    scale: linear
    center: 18.0
    sigma: 3.0
    min: 5.0
    max: 30.0
```

## Posterior Analysis Example

After a fit, the built-in commands can regenerate the standard products:

```bash
dring plot -r results/HD163296_ring1
dring derived -r results/HD163296_ring1
dring intensity -r results/HD163296_ring1
```

For a multi-modal posterior, select one mode in Python before evaluating
derived quantities:

```python
from pathlib import Path

from dring.derived import evaluate_posterior, load_posterior, summarize_table

result_dir = Path("results/HD163296_ring1")
samples = load_posterior(result_dir / "chains" / "equal_weighted_post.txt")

# Parameter columns are usually:
# 0 alpha, 1 vf, 2 eps, 3 T, followed by one calibration factor per band.
mode = (samples[:, 1] > 8.0) & (samples[:, 1] < 80.0)
mode_samples = samples[mode]

derived_table, problem = evaluate_posterior(
    result_dir=result_dir,
    samples=mode_samples,
    tau_wavelength_cm=0.13,
    max_samples=200,
    progress=True,
)

summarize_table(derived_table)
```

Corner-plot display ranges can be set in the YAML file under
`analysis.corner.ranges`.
