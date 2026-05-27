"""UltraNest-facing fitting code."""

from pathlib import Path
import copy
import math
import os

import numpy as np
from scipy.special import ndtri

from . import constants as c
from .data import (
    fit_radii_by_band,
    infer_project_root,
    load_observation_bands,
    normalize_opacity_data,
)
from .model import DustRingModel


def mpi_rank():
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


def write_run_config(config, output_dir):
    """Save the resolved config next to the UltraNest results."""
    import yaml

    config_to_write = _yaml_safe(copy.deepcopy(config))
    project_root = infer_project_root(output_dir)
    for band in config_to_write.get("data", {}).get("bands", []):
        value = band.get("profile_path")
        if value is not None:
            path_value = Path(value).expanduser()
            if path_value.is_absolute():
                try:
                    band["profile_path"] = str(path_value.relative_to(project_root))
                except ValueError:
                    pass

    for key in ("size_opac", "lam_opac", "k_abs_opac", "k_sca_opac", "g_sca_opac", "rhos_opac"):
        value = config_to_write.get("opacity", {}).get(key)
        if isinstance(value, str):
            path_value = Path(value).expanduser()
            if path_value.is_absolute():
                try:
                    config_to_write["opacity"][key] = str(path_value.relative_to(project_root))
                except ValueError:
                    pass

    if config_to_write.get("output_dir") is not None:
        path_value = Path(config_to_write["output_dir"]).expanduser()
        if path_value.is_absolute():
            try:
                config_to_write["output_dir"] = str(path_value.relative_to(project_root))
            except ValueError:
                pass

    path = Path(output_dir) / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_write, f, sort_keys=False)
    return path


def _yaml_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _yaml_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(val) for val in value]
    return value


def _log_uniform(u, lo, hi):
    lo = float(lo)
    hi = float(hi)
    return 10 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo)))


def _linear(u, lo, hi):
    lo = float(lo)
    hi = float(hi)
    return lo + u * (hi - lo)


def _as_prior_dict(spec):
    if isinstance(spec, dict):
        return dict(spec)
    raise ValueError("prior entries must be dicts with type, scale, min, and max")


def _bounded_gaussian(u, center, sigma, lo, hi):
    center = float(center)
    sigma = float(sigma)
    lo = float(lo)
    hi = float(hi)
    if sigma <= 0:
        raise ValueError("gaussian prior sigma must be positive")
    cdf_lo = 0.5 * (1.0 + math.erf((lo - center) / (sigma * np.sqrt(2.0))))
    cdf_hi = 0.5 * (1.0 + math.erf((hi - center) / (sigma * np.sqrt(2.0))))
    if not cdf_hi > cdf_lo:
        raise ValueError("gaussian prior bounds must enclose non-zero probability")
    return center + sigma * ndtri(cdf_lo + u * (cdf_hi - cdf_lo))


def _parameter_prior_transform(u, spec):
    spec = _as_prior_dict(spec)
    mode = str(spec.get("type", spec.get("kind", "flat"))).lower()
    scale = str(spec.get("scale", "linear")).lower()
    if "min" not in spec or "max" not in spec:
        raise ValueError("each prior must define min and max")

    lo = float(spec["min"])
    hi = float(spec["max"])
    if hi <= lo:
        raise ValueError("prior max must be larger than min")

    if scale == "log":
        if lo <= 0 or hi <= 0:
            raise ValueError("log-scale priors require positive min and max")
        lo_t = np.log10(lo)
        hi_t = np.log10(hi)
        if mode == "flat":
            return 10 ** _linear(u, lo_t, hi_t)
        if mode == "gaussian":
            if "center" not in spec or "sigma" not in spec:
                raise ValueError("gaussian priors must define center and sigma")
            center = np.log10(float(spec["center"]))
            sigma = float(spec["sigma"])
            sigma_t = np.log10(10 ** center + sigma) - center if spec.get("sigma_is_linear", True) else sigma
            return 10 ** _bounded_gaussian(u, center, sigma_t, lo_t, hi_t)
    elif scale == "linear":
        if mode == "flat":
            return _linear(u, lo, hi)
        if mode == "gaussian":
            if "center" not in spec or "sigma" not in spec:
                raise ValueError("gaussian priors must define center and sigma")
            return _bounded_gaussian(u, spec["center"], spec["sigma"], lo, hi)
    else:
        raise ValueError("prior scale must be 'linear' or 'log'")

    raise ValueError("prior type must be 'flat' or 'gaussian'")


def _calibration_prior_transform(u, calerr, config):
    mode = config.get("calibration_prior", "flat")
    if mode == "flat":
        nsigma = float(config.get("calibration_prior_nsigma", 3.0))
        lower = 1.0 - nsigma * calerr
        upper = 1.0 + nsigma * calerr
        lower = max(lower, float(config.get("calibration_prior_min", 1e-6)))
        return _linear(u, lower, upper)
    if mode == "gaussian":
        return 1.0 + calerr * ndtri(u)
    raise ValueError("calibration_prior must be 'flat' or 'gaussian'")


def _local_spacing(values):
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.array([np.inf])

    order = np.argsort(values)
    sorted_values = values[order]
    spacing_sorted = np.empty_like(sorted_values)
    spacing_sorted[0] = sorted_values[1] - sorted_values[0]
    spacing_sorted[-1] = sorted_values[-1] - sorted_values[-2]
    if values.size > 2:
        spacing_sorted[1:-1] = 0.5 * (sorted_values[2:] - sorted_values[:-2])

    spacing = np.empty_like(spacing_sorted)
    spacing[order] = spacing_sorted
    return spacing


def _convolution_matrix(r_au, beam_fwhm_au, nsigma=5.0):
    """Return the linear operator equivalent to convolve_radial_profile()."""
    r_au = np.asarray(r_au, dtype=float)
    sigma_b = float(beam_fwhm_au) / 2.355
    if sigma_b <= 0.0:
        return np.eye(r_au.size)

    dr = np.empty_like(r_au)
    dr[1:-1] = 0.5 * (r_au[2:] - r_au[:-2])
    dr[0] = r_au[1] - r_au[0]
    dr[-1] = r_au[-1] - r_au[-2]

    matrix = np.zeros((r_au.size, r_au.size), dtype=float)
    for i in range(r_au.size):
        d = r_au - r_au[i]
        mask = np.abs(d) <= nsigma * sigma_b
        if not np.any(mask):
            matrix[i, i] = 1.0
            continue
        weights = np.exp(-0.5 * (d[mask] / sigma_b) ** 2) * dr[mask]
        weights /= weights.sum()
        matrix[i, mask] = weights
    return matrix


def _interpolation_matrix(x_new, x_grid):
    """Return the linear operator equivalent to np.interp(x_new, x_grid, y)."""
    x_new = np.asarray(x_new, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    matrix = np.zeros((x_new.size, x_grid.size), dtype=float)

    for i, x in enumerate(x_new):
        if x <= x_grid[0]:
            matrix[i, 0] = 1.0
        elif x >= x_grid[-1]:
            matrix[i, -1] = 1.0
        else:
            hi = np.searchsorted(x_grid, x)
            lo = hi - 1
            weight_hi = (x - x_grid[lo]) / (x_grid[hi] - x_grid[lo])
            matrix[i, lo] = 1.0 - weight_hi
            matrix[i, hi] = weight_hi
    return matrix


class RingFitProblem:
    def __init__(self, config):
        self.config = config
        observations = load_observation_bands(config)
        self.radius_au = observations["radius_au"]
        self.obs_intensity = observations["intensity"]
        self.obs_rms = observations["rms_error"]
        self.wavelengths = observations["wavelengths"]
        self.band_names = observations["band_names"]
        self.calerr = observations["calerr"]
        self.n_band = len(self.wavelengths)
        if self.calerr.size != self.n_band:
            raise ValueError("each data.bands entry must provide one cal_error")
        if len(self.obs_intensity) != self.n_band:
            raise ValueError("number of data.bands entries must match wavelength count")

        self.distance_pc = float(config.get("distance_pc", 100.0))
        self.inclination_deg = float(config.get("inclination_deg", 0.0))
        self.beams_arcsec = observations["angular_resolution_arcsec"]
        if config.get("beam_deprojection", False):
            cosi = np.cos(np.deg2rad(self.inclination_deg))
            if cosi <= 0:
                raise ValueError("inclination_deg must be < 90 when beam_deprojection is true")
            self.beams_arcsec = self.beams_arcsec / np.sqrt(cosi)
        self.beams_au = self.beams_arcsec * self.distance_pc
        self.fit_radii = fit_radii_by_band(config, self.n_band)
        self.fit_indices = [
            np.array([np.argmin(np.abs(self.radius_au[i] - r)) for r in radii])
            for i, radii in enumerate(self.fit_radii)
        ]

        opacity_cfg = config["opacity"]
        self.opacity = normalize_opacity_data(opacity_cfg)
        self.model_cfg = config["model"]
        self.model_r_grid_au = self._load_model_r_grid_au()
        self.model_size = np.logspace(
            np.log10(float(self.model_cfg.get("a_min", 1e-5))),
            np.log10(float(self.model_cfg.get("a_max", 100.0))),
            int(self.model_cfg.get("size_res", 140)),
        )
        self.model_grid_cache = DustRingModel.build_grid_cache(
            self.model_r_grid_au * c.au,
            float(self.model_cfg.get("a_min", 1e-5)),
            float(self.model_cfg.get("a_max", 100.0)),
            self.opacity["rho_s"],
            int(self.model_cfg.get("size_res", 140)),
            float(self.model_cfg["ring_center_au"]) * c.au,
            float(self.model_cfg["stellar_mass_msun"]) * c.M_sun,
            float(self.model_cfg["pressure_width_au"]) * c.au,
        )
        self.opacity_at_model_size = self._precompute_opacity_at_model_size()
        self.convolution_matrices = [
            _convolution_matrix(self.model_r_grid_au, beam_au) for beam_au in self.beams_au
        ]
        self._precompute_fit_data()
        self.fit_projection_matrices = [
            _interpolation_matrix(self.fit_radii_au[i], self.model_r_grid_au)
            @ self.convolution_matrices[i]
            for i in range(self.n_band)
        ]

        self.parameters = ["alpha", "vf", "eps", "T"] + [
            f"c_lam{i + 1}" for i in range(self.n_band)
        ]

    def _precompute_fit_data(self):
        self.fit_radii_au = []
        self.fit_obs_intensity = []
        self.fit_rms = []
        self.fit_oversampling = []
        self.fit_rms_eff = []
        self.fit_ndata_raw = 0
        self.fit_ndata_eff = 0.0

        for iband, idx in enumerate(self.fit_indices):
            radii_au = self.radius_au[iband][idx]
            oversampling = self.oversampling_factors(iband, radii_au)

            self.fit_radii_au.append(radii_au)
            self.fit_obs_intensity.append(self.obs_intensity[iband][idx])
            self.fit_rms.append(self.obs_rms[iband][idx])
            self.fit_oversampling.append(oversampling)
            self.fit_rms_eff.append(self.obs_rms[iband][idx] * np.sqrt(oversampling))
            self.fit_ndata_raw += int(idx.size)
            self.fit_ndata_eff += float(np.sum(1.0 / oversampling))

    def _load_model_r_grid_au(self):
        r_grid_cfg = self.model_cfg.get("r_grid_au")
        if r_grid_cfg not in (None, "auto"):
            return np.linspace(float(r_grid_cfg[0]), float(r_grid_cfg[1]), int(r_grid_cfg[2]))

        fit_radii = np.concatenate([np.asarray(r, dtype=float) for r in self.fit_radii])
        ring_center = float(self.model_cfg["ring_center_au"])
        beam_margin = float(self.model_cfg.get("r_grid_beam_margin", 5.0)) * float(
            np.max(self.beams_au)
        )
        extra_margin = float(self.model_cfg.get("r_grid_extra_margin_au", 0.0))
        margin = beam_margin + extra_margin

        r_min = min(float(np.min(fit_radii)), ring_center) - margin
        r_max = max(float(np.max(fit_radii)), ring_center) + margin
        r_min_floor = float(self.model_cfg.get("r_grid_min_radius_au", 1e-3))
        r_min = max(r_min_floor, r_min)
        if "r_grid_n" in self.model_cfg or "r_grid_res" in self.model_cfg:
            n_grid = int(self.model_cfg.get("r_grid_n", self.model_cfg.get("r_grid_res")))
        else:
            min_dr_au = float(np.min(self.beams_au)) / float(
                self.model_cfg.get("r_grid_beam_points", 5.0)
            )
            min_dr_au = min(
                min_dr_au,
                float(self.model_cfg["pressure_width_au"])
                / float(self.model_cfg.get("r_grid_width_points", 8.0)),
                float(self.model_cfg.get("r_grid_max_dr_au", 1.0)),
            )
            min_dr_au = max(min_dr_au, float(self.model_cfg.get("r_grid_min_dr_au", 0.05)))
            n_grid = int(np.ceil((r_max - r_min) / min_dr_au)) + 1
            n_grid = max(n_grid, int(self.model_cfg.get("r_grid_min_n", 80)))
            n_grid = min(n_grid, int(self.model_cfg.get("r_grid_max_n", 300)))
        if n_grid < 3:
            raise ValueError("model.r_grid_n must be >= 3")
        if r_max <= r_min:
            raise ValueError("automatic model radial grid has non-positive width")
        return np.linspace(r_min, r_max, n_grid)

    def _precompute_opacity_at_model_size(self):
        opacity_a = np.asarray(self.opacity["a"], dtype=float)
        opacity_lam = np.asarray(self.opacity["lam"], dtype=float)
        k_abs_table = np.asarray(self.opacity["k_abs"], dtype=float)
        k_sca_table = np.asarray(self.opacity["k_sca"], dtype=float)
        g_table = np.asarray(self.opacity["g"], dtype=float)

        k_abs_size = np.empty((self.n_band, self.model_size.size), dtype=float)
        k_sca_eff_size = np.empty_like(k_abs_size)

        for iwv, wav in enumerate(self.wavelengths):
            k_abs_at_wav = np.array(
                [np.interp(wav, opacity_lam, k_abs_table[ia, :]) for ia in range(len(opacity_a))]
            )
            k_sca_at_wav = np.array(
                [np.interp(wav, opacity_lam, k_sca_table[ia, :]) for ia in range(len(opacity_a))]
            )
            g_at_wav = np.array(
                [np.interp(wav, opacity_lam, g_table[ia, :]) for ia in range(len(opacity_a))]
            )
            k_abs_size[iwv] = np.interp(self.model_size, opacity_a, k_abs_at_wav)
            k_sca_eff_size[iwv] = np.interp(
                self.model_size, opacity_a, (1.0 - g_at_wav) * k_sca_at_wav
            )

        return {
            "k_abs": k_abs_size,
            "k_sca_eff": k_sca_eff_size,
        }

    def oversampling_factors(self, iband, radii_au):
        if not self.config.get("oversampling_correction", True):
            return np.ones_like(radii_au, dtype=float)

        spacing_au = _local_spacing(radii_au)
        beam_au = self.beams_arcsec[iband] * self.distance_pc
        factors = np.maximum(1.0, beam_au / spacing_au)
        return factors

    def prior_transform(self, cube):
        p = np.asarray(cube, dtype=float).copy()
        prior = self.config["priors"]
        p[0] = _parameter_prior_transform(cube[0], prior["alpha"])
        p[1] = _parameter_prior_transform(cube[1], prior["vf"])
        p[2] = _parameter_prior_transform(cube[2], prior["eps"])
        p[3] = _parameter_prior_transform(cube[3], prior["T"])
        for i in range(self.n_band):
            p[4 + i] = _calibration_prior_transform(cube[4 + i], self.calerr[i], self.config)
        return p

    def make_ring_model(self, theta):
        """Build and run the ring model for a sampled parameter vector."""
        alpha, vf, eps, temp = theta[:4]
        model_cfg = self.model_cfg
        ring = DustRingModel(
            alpha,
            vf,
            float(model_cfg["sigma_g"]),
            eps,
            float(model_cfg["pressure_width_au"]) * c.au,
            temp,
            R_0=float(model_cfg["ring_center_au"]) * c.au,
            r_grid=self.model_r_grid_au * c.au,
            M_star=float(model_cfg["stellar_mass_msun"]) * c.M_sun,
            rho_s=self.opacity["rho_s"],
            size_res=int(model_cfg.get("size_res", 140)),
            a_min=float(model_cfg.get("a_min", 1e-5)),
            a_max=float(model_cfg.get("a_max", 100.0)),
            T_slope=float(model_cfg.get("temperature_slope", -0.5)),
            q=float(model_cfg.get("size_distribution_q", -3.5)),
            grid_cache=self.model_grid_cache,
        )
        ring.run_model()
        return ring

    def _make_ring_model(self, theta):
        return self.make_ring_model(theta)

    def raw_model_profiles(self, theta):
        ring = self.make_ring_model(theta)
        raw = DustRingModel.scattering_precomputed(
            self.wavelengths,
            ring.SigmaDust,
            ring.T,
            self.inclination_deg,
            self.opacity_at_model_size["k_abs"],
            self.opacity_at_model_size["k_sca_eff"],
        )
        return self.model_r_grid_au, raw

    def model_profiles(self, theta):
        model_r_au, raw = self.raw_model_profiles(theta)

        profiles = []
        for i in range(self.n_band):
            profiles.append(self.convolution_matrices[i] @ raw[i])
        return model_r_au, np.asarray(profiles)

    def log_likelihood(self, params):
        cal_factors = np.asarray(params[4 : 4 + self.n_band])
        try:
            _, raw = self.raw_model_profiles(params)
        except (FloatingPointError, ZeroDivisionError, OverflowError, ValueError):
            return -np.inf
        if not np.all(np.isfinite(raw)):
            return -np.inf
        loglike = 0.0
        for iband in range(self.n_band):
            model_at_data = self.fit_projection_matrices[iband] @ raw[iband]
            model_at_data = cal_factors[iband] * model_at_data
            residual = (self.fit_obs_intensity[iband] - model_at_data) / self.fit_rms_eff[iband]
            loglike += -0.5 * np.sum(residual**2)
            if self.config.get("include_calibration_penalty", True):
                loglike += -0.5 * ((cal_factors[iband] - 1.0) / self.calerr[iband]) ** 2
        return loglike


def run_fit(config):
    from ultranest import ReactiveNestedSampler

    problem = RingFitProblem(config)
    sampler_cfg = config.get("sampler", {})
    output_dir = Path(config.get("output_dir", "results/run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if mpi_rank() == 0:
        write_run_config(config, output_dir)
    sampler = ReactiveNestedSampler(
        problem.parameters,
        problem.log_likelihood,
        problem.prior_transform,
        log_dir=str(output_dir),
        resume=config.get("resume", "overwrite"),
    )

    stepsampler = sampler_cfg.get("stepsampler", config.get("stepsampler"))
    if stepsampler is not None:
        stepsampler = str(stepsampler).lower()
    if stepsampler in ("slice", "slicesampler"):
        import ultranest.stepsampler

        nsteps = int(
            sampler_cfg.get(
                "nsteps",
                sampler_cfg.get("slice_steps", config.get("slice_steps", 2 * len(problem.parameters))),
            )
        )
        sampler.stepsampler = ultranest.stepsampler.SliceSampler(
            nsteps=nsteps,
            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
        )
    elif stepsampler not in (None, "none", "false"):
        raise ValueError("sampler.stepsampler must be 'slice' or omitted")

    results = sampler.run(
        min_num_live_points=int(sampler_cfg.get("min_num_live_points", config.get("min_num_live_points", 200))),
        frac_remain=float(sampler_cfg.get("frac_remain", config.get("frac_remain", 0.01))),
    )
    if getattr(sampler, "stepsampler", None) is not None and mpi_rank() == 0:
        sampler.stepsampler.plot(filename=str(output_dir / "stepsampler_stats.pdf"))
    if config.get("make_plots", True) and mpi_rank() == 0:
        sampler.plot_run()
        sampler.plot_trace()
        sampler.plot_corner()
    return results, problem
