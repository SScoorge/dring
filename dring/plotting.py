"""Packaged plotting helpers for installed ``dring`` users."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dring-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from ringfit.data import load_config
from ringfit.fitter import RingFitProblem, mpi_rank

from .corner_tools import corner_like_plot, summary_stats


DEFAULT_CORNER_LABELS = [
    r"$\log(\alpha)$",
    r"$\log(v_{\rm frag})\,[{\rm cm\,s^{-1}}]$",
    r"$\log(\epsilon_0)$",
    r"$T_0\,[{\rm K}]$",
]


def _load_samples(path: Path) -> np.ndarray:
    try:
        samples = np.loadtxt(path)
    except ValueError:
        samples = np.loadtxt(path, skiprows=1)
    return np.atleast_2d(samples)


def _read_result_config(result_dir: Path) -> dict:
    config_path = result_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"result config not found: {config_path}")
    return load_config(config_path)


def _analysis_section(config: dict, name: str) -> dict:
    return dict(config.get("analysis", {}).get(name, {}))


def _dimension_key_map(labels, names):
    keys = {}
    for i, name in enumerate(names):
        keys[str(i)] = i
        keys[i] = i
        keys[name] = i
        keys[name.lower()] = i
        if i < len(labels):
            keys[labels[i]] = i
    return keys


def _parse_plot_ranges(value, labels, names, log_flags=None, transform_log=False):
    def _coerce(idx, limits):
        lo, hi = tuple(limits)
        lo = float(lo)
        hi = float(hi)
        if transform_log and log_flags is not None and log_flags[idx]:
            if lo <= 0 or hi <= 0:
                raise ValueError(f"log range for {names[idx]} must be positive")
            lo, hi = np.log10(lo), np.log10(hi)
        return lo, hi

    if value in (None, "auto"):
        return None
    if isinstance(value, (list, tuple)):
        return {i: _coerce(i, v) for i, v in enumerate(value) if v is not None}
    if isinstance(value, dict):
        key_map = _dimension_key_map(labels, names)
        parsed = {}
        for key, limits in value.items():
            idx = key_map.get(key, key_map.get(str(key).lower()))
            if idx is None:
                raise ValueError(f"unknown corner range key: {key}")
            parsed[idx] = _coerce(idx, limits)
        return parsed
    raise ValueError("corner ranges must be a list, dict, or 'auto'")


def _apply_axis_limits(axes, limits, *, axis, labels, names, log_flags):
    parsed = (
        _parse_plot_ranges(limits, labels, names, log_flags=log_flags, transform_log=True)
        if limits
        else None
    )
    if not parsed:
        return
    for idx, value in parsed.items():
        lo, hi = value
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                ax = axes[i, j]
                if not ax.get_visible():
                    continue
                if axis == "x" and j == idx:
                    ax.set_xlim(lo, hi)
                if axis == "y" and i == idx and i != j:
                    ax.set_ylim(lo, hi)


def plot_result(result_dir, *, output=None, columns=None, labels=None, ranges=None, xlim=None, ylim=None):
    """Plot the default corner figure for a result directory."""
    if mpi_rank() != 0:
        return None
    result_dir = Path(result_dir)
    config = _read_result_config(result_dir)
    corner_cfg = _analysis_section(config, "corner")
    posterior = result_dir / "chains" / "equal_weighted_post.txt"
    if not posterior.exists():
        raise FileNotFoundError(f"posterior samples not found: {posterior}")

    samples = _load_samples(posterior)
    if columns is None:
        columns = corner_cfg.get("columns", list(range(min(4, samples.shape[1]))))
    samples = samples[:, columns]
    n_dim = samples.shape[1]

    if labels is None:
        labels = corner_cfg.get("labels", DEFAULT_CORNER_LABELS[:n_dim])
    if len(labels) != n_dim:
        raise ValueError("corner labels length must match selected columns")
    log_flags = corner_cfg.get("log_flags")
    if log_flags is None:
        log_flags = ["log" in str(label).lower() for label in labels]

    param_names = ["alpha", "vf", "eps", "T"]
    param_names.extend(f"c_lam{i + 1}" for i in range(max(0, max(columns) - 3)))
    names = [param_names[i] if i < len(param_names) else str(i) for i in columns]
    if ranges is None:
        ranges = corner_cfg.get("ranges", corner_cfg.get("plot_ranges"))
    if xlim is None:
        xlim = corner_cfg.get("xlim")
    if ylim is None:
        ylim = corner_cfg.get("ylim")

    show_diagonal_stats = bool(corner_cfg.get("show_diagonal_stats", False))
    stats = summary_stats(samples) if show_diagonal_stats else None
    fig, axes = corner_like_plot(
        samples,
        labels=labels,
        log_flags=log_flags,
        bins=int(corner_cfg.get("bins", 30)),
        kde_bw=float(corner_cfg.get("kde_bw", 0.2)),
        plot_ranges=_parse_plot_ranges(ranges, labels, names),
        stats1=stats,
        stats2=None,
    )
    _apply_axis_limits(axes, xlim, axis="x", labels=labels, names=names, log_flags=log_flags)
    _apply_axis_limits(axes, ylim, axis="y", labels=labels, names=names, log_flags=log_flags)

    if output is None:
        output = corner_cfg.get("output", result_dir / "corner.png")
    output = Path(output)
    if not output.is_absolute():
        output = result_dir / output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def _params_from_results(results, problem):
    if results is None or not isinstance(results, dict):
        return None
    for key in ("maximum_likelihood", "maximum_a_posteriori"):
        value = results.get(key)
        if isinstance(value, dict):
            point = value.get("point")
            if point is not None:
                return np.asarray(point, dtype=float)[: len(problem.parameters)]
    posterior = results.get("posterior")
    if isinstance(posterior, dict):
        for key in ("median", "mean"):
            point = posterior.get(key)
            if point is not None:
                return np.asarray(point, dtype=float)[: len(problem.parameters)]
    return None


def _params_from_posterior(result_dir, problem):
    posterior = result_dir / "chains" / "equal_weighted_post.txt"
    samples = _load_samples(posterior)
    return np.nanmedian(samples[:, : len(problem.parameters)], axis=0)


def _band_colors(n_band, config):
    cmap_name = config.get("cmap", "viridis")
    cmap = plt.get_cmap(cmap_name)
    if n_band == 1:
        positions = np.array([0.5])
    else:
        lo, hi = config.get("cmap_range", (0.08, 0.92))
        positions = np.linspace(float(lo), float(hi), int(n_band))
    return [cmap(x) for x in positions]


def plot_intensity_fit(result_dir, *, params=None, results=None, output=None, xlim=None, ylim=None, title=None):
    """Plot observed intensity profiles and the calibrated best-fit model."""
    if mpi_rank() != 0:
        return None
    result_dir = Path(result_dir)
    config = _read_result_config(result_dir)
    intensity_cfg = _analysis_section(config, "intensity")
    problem = RingFitProblem(config)

    if params is None:
        params = _params_from_results(results, problem)
    if params is None:
        params = _params_from_posterior(result_dir, problem)
    params = np.asarray(params, dtype=float)[: len(problem.parameters)]

    model_r_au, model_profiles = problem.model_profiles(params)
    cal_factors = np.asarray(params[4 : 4 + problem.n_band], dtype=float)
    colors = _band_colors(problem.n_band, intensity_cfg)

    fig, ax = plt.subplots(
        figsize=tuple(intensity_cfg.get("figsize", (9.5, 6.0))),
        dpi=int(intensity_cfg.get("dpi", 180)),
    )

    all_radii = []
    fit_positive_values = []
    all_positive_values = []
    for iband in range(problem.n_band):
        radius = np.asarray(problem.radius_au[iband], dtype=float)
        intensity = np.asarray(problem.obs_intensity[iband], dtype=float)
        rms = np.asarray(problem.obs_rms[iband], dtype=float)

        finite = np.isfinite(radius) & np.isfinite(intensity) & np.isfinite(rms)
        positive = finite & (intensity > 0)
        if np.any(finite):
            all_radii.append(radius[finite])
        if np.any(positive):
            all_positive_values.append(intensity[positive])
            y_floor = np.nanmin(intensity[positive]) * 1e-3
        else:
            y_floor = 1e-99

        fit_indices = np.asarray(problem.fit_indices[iband], dtype=int)
        fit_values = intensity[fit_indices]
        fit_positive = np.isfinite(fit_values) & (fit_values > 0)
        if np.any(fit_positive):
            fit_positive_values.append(fit_values[fit_positive])

        y_low = np.maximum(intensity - rms, y_floor)
        y_high = intensity + rms
        fill_mask = finite & (y_high > 0)
        band_name = getattr(problem, "band_names", [])[iband] if hasattr(problem, "band_names") else None
        if band_name:
            lam_label = f"{band_name} ({problem.wavelengths[iband] * 10.0:.3g} mm)"
        else:
            lam_label = f"{problem.wavelengths[iband] * 10.0:.3g} mm"
        color = colors[iband]

        ax.plot(radius[finite], intensity[finite], color=color, lw=2.0, alpha=0.78, label=f"data {lam_label}")
        ax.fill_between(radius[fill_mask], y_low[fill_mask], y_high[fill_mask], color=color, alpha=0.14, lw=0)
        ax.plot(
            model_r_au,
            cal_factors[iband] * model_profiles[iband],
            color=color,
            lw=2.8,
            ls="--",
            label=f"model {lam_label}",
        )
        ax.scatter(
            radius[fit_indices],
            intensity[fit_indices],
            color=color,
            s=42,
            edgecolor="k",
            linewidth=0.45,
            zorder=5,
        )

    ax.set_yscale("log")
    if xlim is None:
        xlim = intensity_cfg.get("xlim")
    if ylim is None:
        ylim = intensity_cfg.get("ylim")
    if xlim is None and all_radii:
        radii = np.concatenate(all_radii)
        ax.set_xlim(float(np.nanmin(radii)), float(np.nanmax(radii)))
    elif xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is None:
        values = np.concatenate(fit_positive_values) if fit_positive_values else np.array([])
        if values.size == 0 and all_positive_values:
            values = np.concatenate(all_positive_values)
        if values.size:
            ax.set_ylim(float(np.nanmin(values)) / 100.0, float(np.nanmax(values)) * 10.0)
    elif ylim != "auto":
        ax.set_ylim(*ylim)

    ax.set_xlabel(r"$r\ \mathrm{[au]}$")
    ax.set_ylabel(r"$I_\lambda\ \mathrm{[Jy\ sr^{-1}]}$")
    if title is None:
        title = intensity_cfg.get("title", f"Intensity profile and calibrated model: {result_dir.name}")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=intensity_cfg.get("legend_fontsize", 11), ncols=intensity_cfg.get("legend_ncols", 2))
    ax.tick_params(direction="in", which="both", top=True, right=True)
    fig.tight_layout()

    if output is None:
        output = intensity_cfg.get("output", result_dir / "intensity_bestfit.png")
    output = Path(output)
    if not output.is_absolute():
        output = result_dir / output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output
