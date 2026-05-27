"""Stable Python-facing helpers for fitting, plotting, and model debugging."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ringfit import constants as c
from ringfit.data import load_config as _load_config
from ringfit.data import normalize_opacity_data
from ringfit.model import DustRingModel

__version__ = "0.1.0"


def _missing_extra_error(extra: str, exc: Exception) -> RuntimeError:
    err = RuntimeError(
        f"This command requires optional fitting/analysis dependencies. "
        f"Install them with: pip install 'dring[{extra}]'"
    )
    err.__cause__ = exc
    return err


def _mpi_rank() -> int:
    import os

    mpi_env_vars = (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_RANK",
        "MPI_LOCALNRANKS",
        "MV2_COMM_WORLD_SIZE",
    )
    if not any(name in os.environ for name in mpi_env_vars):
        return 0
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_rank()
    except Exception:
        return 0


def _load_fitter():
    try:
        from ringfit.fitter import RingFitProblem, run_fit
    except ImportError as exc:
        raise _missing_extra_error("fit", exc)
    return RingFitProblem, run_fit


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and resolve paths relative to that config."""
    return _load_config(path)


def _config_from_any(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    return load_config(config)


def fit(config: str | Path | Mapping[str, Any], *, check_config: bool = False):
    """Run a fit from a config path or dict.

    When ``check_config`` is true, inputs are loaded and a ``RingFitProblem`` is
    returned without starting UltraNest.
    """
    cfg = _config_from_any(config)
    RingFitProblem, run_fit = _load_fitter()
    if check_config:
        return RingFitProblem(cfg)
    results, problem = run_fit(cfg)
    if _mpi_rank() == 0:
        _run_post_fit_analysis(cfg, results)
    return results, problem


def _run_post_fit_analysis(config: Mapping[str, Any], results) -> None:
    analysis_cfg = dict(config.get("analysis", {}))
    output_dir = Path(config.get("output_dir", "results/run"))

    if analysis_cfg.get("enabled", True) is False:
        return

    if dict(analysis_cfg.get("corner", {})).get("enabled", True):
        from .plotting import plot_result

        path = plot_result(output_dir)
        print(f"wrote {path}")

    if dict(analysis_cfg.get("derived", {})).get("enabled", True):
        derived_cfg = dict(analysis_cfg.get("derived", {}))
        max_samples = derived_cfg["max_samples"] if "max_samples" in derived_cfg else 100
        derived_table(
            output_dir,
            max_samples=max_samples,
            progress=bool(derived_cfg.get("progress", True)),
            output=derived_cfg.get("output", "derived_summary.txt"),
        )

    if dict(analysis_cfg.get("intensity", {})).get("enabled", True):
        from .plotting import plot_intensity_fit

        path = plot_intensity_fit(output_dir, results=results)
        print(f"wrote {path}")


def demo(*, run: bool = False, output_dir: str | Path | None = None):
    """Run the packaged demo check, or a tiny demo fit when ``run=True``."""
    config_path = resources.files("dring").joinpath("examples/demo.yaml")
    cfg = load_config(config_path)
    example_dir = Path(str(resources.files("dring").joinpath("examples")))
    for band in cfg["data"]["bands"]:
        band["profile_path"] = str(example_dir / Path(band["profile_path"]).name)
    for key, value in cfg["opacity"].items():
        path_value = Path(value)
        if path_value.parent.name:
            cfg["opacity"][key] = str(example_dir / path_value.parent.name / path_value.name)
        else:
            cfg["opacity"][key] = str(example_dir / path_value.name)
    if output_dir is not None:
        cfg["output_dir"] = str(output_dir)
    if run:
        return fit(cfg)

    RingFitProblem, _ = _load_fitter()
    problem = RingFitProblem(cfg)
    cube = np.full(len(problem.parameters), 0.5)
    params = problem.prior_transform(cube)
    r_au, profiles = problem.model_profiles(params)
    return {
        "config": cfg,
        "problem": problem,
        "parameters": problem.parameters,
        "theta": params,
        "r_au": r_au,
        "profiles": profiles,
    }


def make_model(
    *,
    alpha: float = 1e-3,
    vf: float = 100.0,
    sigma_g: float = 10.0,
    eps: float = 1e-2,
    pressure_width_au: float = 10.0,
    temperature: float = 20.0,
    ring_center_au: float = 100.0,
    stellar_mass_msun: float = 1.0,
    r_grid_au: Any = None,
    a_min: float = 1e-5,
    a_max: float = 100.0,
    rho_s: float = 1.67,
    size_res: int = 140,
    temperature_slope: float = -0.5,
    size_distribution_q: float = -3.5,
    xi: float = 11 / 6,
    grid_cache: Mapping[str, Any] | None = None,
    run: bool = True,
) -> DustRingModel:
    """Create a dust-ring model with user-facing units.

    Radius-like arguments are in au, stellar mass is in solar masses, and all
    grain sizes remain in cm to match the opacity tables.
    """
    if r_grid_au is None:
        r_min = max(1e-3, ring_center_au - 5 * pressure_width_au)
        r_grid_au = np.linspace(r_min, ring_center_au + 5 * pressure_width_au, 300)

    model = DustRingModel(
        alpha,
        vf,
        sigma_g,
        eps,
        pressure_width_au * c.au,
        temperature,
        xi=xi,
        R_0=ring_center_au * c.au,
        M_star=stellar_mass_msun * c.M_sun,
        r_grid=np.asarray(r_grid_au, dtype=float) * c.au,
        a_min=a_min,
        a_max=a_max,
        rho_s=rho_s,
        size_res=size_res,
        T_slope=temperature_slope,
        q=size_distribution_q,
        grid_cache=grid_cache,
    )
    if run:
        model.run_model()
    return model


def compute_profile(
    params: Any | None = None,
    config: str | Path | Mapping[str, Any] | None = None,
    *,
    model: DustRingModel | None = None,
    wavelengths_cm: Any = None,
    opacity: Mapping[str, Any] | None = None,
    opacity_path: str | Path | None = None,
    inclination_deg: float = 0.0,
):
    """Compute model intensity profiles.

    Pass ``config`` plus sampled ``params`` to use the full fitting projection,
    or pass a pre-built ``model`` with wavelengths and opacity for external
    MCMC/radiative-transfer workflows.
    """
    if config is not None:
        RingFitProblem, _ = _load_fitter()
        problem = RingFitProblem(_config_from_any(config))
        if params is None:
            params = problem.prior_transform(np.full(len(problem.parameters), 0.5))
        return problem.raw_model_profiles(params)

    if model is None:
        model = make_model()
    if not hasattr(model, "SigmaDust"):
        model.run_model()
    if opacity is None:
        if opacity_path is None:
            raise ValueError("opacity is required when config is not supplied")
        raise ValueError("opacity_path is no longer supported; pass an opacity dict with size_opac, lam_opac, k_abs_opac, k_sca_opac, g_sca_opac, and rhos_opac")
    else:
        opacity = normalize_opacity_data(opacity)
    if wavelengths_cm is None:
        raise ValueError("wavelengths_cm is required when config is not supplied")

    wavelengths_cm = np.asarray(wavelengths_cm, dtype=float)
    opacity_a = np.asarray(opacity["a"], dtype=float)
    opacity_lam = np.asarray(opacity["lam"], dtype=float)
    k_abs = np.empty((wavelengths_cm.size, model.size.size), dtype=float)
    k_sca_eff = np.empty_like(k_abs)

    for iwv, wav in enumerate(wavelengths_cm):
        k_abs_at_wav = np.array([np.interp(wav, opacity_lam, row) for row in opacity["k_abs"]])
        k_sca_at_wav = np.array([np.interp(wav, opacity_lam, row) for row in opacity["k_sca"]])
        g_at_wav = np.array([np.interp(wav, opacity_lam, row) for row in opacity["g"]])
        k_abs[iwv] = np.interp(model.size, opacity_a, k_abs_at_wav)
        k_sca_eff[iwv] = np.interp(model.size, opacity_a, (1.0 - g_at_wav) * k_sca_at_wav)

    return model.r_grid / c.au, DustRingModel.scattering_precomputed(
        wavelengths_cm,
        model.SigmaDust,
        model.T,
        inclination_deg,
        k_abs,
        k_sca_eff,
    )


def plot_result(
    result_dir: str | Path,
    *,
    output: str | Path | None = None,
    columns=None,
    labels=None,
    ranges=None,
    xlim=None,
    ylim=None,
):
    """Create the default corner plot for an UltraNest result."""
    from .plotting import plot_result as _plot_result

    return _plot_result(
        result_dir,
        output=output,
        columns=columns,
        labels=labels,
        ranges=ranges,
        xlim=xlim,
        ylim=ylim,
    )


def plot_intensity_fit(
    result_dir: str | Path,
    *,
    params=None,
    results=None,
    output: str | Path | None = None,
    xlim=None,
    ylim=None,
    title=None,
):
    """Plot observed intensities together with the calibrated best-fit model."""
    from .plotting import plot_intensity_fit as _plot_intensity_fit

    return _plot_intensity_fit(
        result_dir,
        params=params,
        results=results,
        output=output,
        xlim=xlim,
        ylim=ylim,
        title=title,
    )


def derived_table(
    result_dir: str | Path,
    *,
    samples=None,
    tau_wavelength_cm: float | None = None,
    max_samples=None,
    stride: int = 1,
    progress: bool = True,
    summarize: bool = True,
    output: str | Path | None = None,
):
    """Evaluate and optionally print the posterior-derived quantity table."""
    if _mpi_rank() != 0:
        return None, None
    try:
        from .derived import evaluate_posterior, summarize_table
    except ImportError as exc:
        raise _missing_extra_error("fit", exc)

    config = load_config(Path(result_dir) / "config.yaml")
    derived_cfg = config.get("analysis", {}).get("derived", {})
    if tau_wavelength_cm is None:
        tau_wavelength_cm = float(derived_cfg.get("tau_wavelength_cm", 0.13))
    if max_samples is None:
        max_samples = derived_cfg.get("max_samples")
    if output is None:
        output = derived_cfg.get("output", "derived_summary.txt")
    if output is not None:
        output = Path(output)
        if not output.is_absolute():
            output = Path(result_dir) / output
    table, problem = evaluate_posterior(
        result_dir=result_dir,
        samples=samples,
        tau_wavelength_cm=tau_wavelength_cm,
        max_samples=max_samples,
        stride=stride,
        progress=progress,
    )
    if summarize:
        summarize_table(table, columns=derived_cfg.get("columns"), output=output)
    return table, problem
