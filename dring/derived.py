"""Posterior-derived physical quantities and LaTeX table rows."""

from pathlib import Path

import numpy as np

from ringfit import constants as c
from ringfit.data import load_config
from ringfit.fitter import RingFitProblem
from ringfit.model import DustRingModel, convolve_radial_profile

try:
    from project_paths import find_project_root
except Exception:
    find_project_root = None


DERIVED_COLUMNS = [
    "alpha",
    "vf",
    "eps",
    "T",
    "a_frag",
    "St0",
    "M_dust",
    "w_eps",
    "tau_abs",
    "chi2_data",
    "red_chi2_data",
    "ndata_raw",
    "ndata_eff",
    "red_chi2_denom",
    "loglike_total",
]


def resolve_result_inputs(config_path, posterior_path=None):
    """Resolve a config path/result directory plus its posterior file.

    ``config_path`` may be either a YAML file or a result directory containing
    ``config.yaml``. When ``posterior_path`` is omitted, the matching
    ``chains/equal_weighted_post.txt`` file is inferred.
    """
    config_path = Path(config_path).expanduser()
    if not config_path.exists() and not config_path.is_absolute() and find_project_root is not None:
        config_path = find_project_root() / config_path
    config_path = config_path.resolve()

    result_dir = None
    if config_path.is_dir():
        result_dir = config_path
        config_path = result_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"could not find {config_path}")

    if posterior_path is None:
        if result_dir is not None:
            posterior_path = result_dir / "chains/equal_weighted_post.txt"
        else:
            config = load_config(config_path)
            posterior_path = Path(config.get("output_dir", "results/run")) / "chains/equal_weighted_post.txt"
    else:
        posterior_path = Path(posterior_path).expanduser()
        if not posterior_path.is_absolute():
            if result_dir is not None:
                posterior_path = result_dir / posterior_path
            elif find_project_root is not None:
                posterior_path = find_project_root() / posterior_path

    return config_path, Path(posterior_path).resolve()


def load_posterior(path, columns=None):
    """Load an UltraNest equal-weight posterior file with or without a header."""
    try:
        arr = np.loadtxt(path)
    except ValueError:
        arr = np.loadtxt(path, skiprows=1)
    arr = np.atleast_2d(arr)
    if columns is not None:
        arr = arr[:, list(columns)]
    return arr


def q1684(values):
    return np.percentile(np.asarray(values, dtype=float), [16, 50, 84])


def median_pm(values):
    q16, q50, q84 = q1684(values)
    return q50, q50 - q16, q84 - q50


def complete_params(problem, sample):
    sample = np.asarray(sample, dtype=float)
    if sample.size >= len(problem.parameters):
        return sample[: len(problem.parameters)]
    if sample.size < 4:
        raise ValueError("posterior sample must contain at least alpha, vf, eps, T")
    params = np.ones(len(problem.parameters), dtype=float)
    params[: sample.size] = sample
    return params


def model_r_grid_au(config):
    r_grid_cfg = config["model"].get("r_grid_au")
    if r_grid_cfg in (None, "auto"):
        problem = RingFitProblem(config)
        return problem.model_r_grid_au.copy()
    lo, hi, n = r_grid_cfg
    return np.linspace(float(lo), float(hi), int(n))


def make_model(problem, params, r_grid_au, eps_scale=1.0):
    alpha, vf, eps, temp = params[:4]
    model_cfg = problem.config["model"]
    grid_cache = getattr(problem, "model_grid_cache", None)
    if grid_cache is not None:
        r_grid_cm = np.asarray(r_grid_au, dtype=float) * c.au
        if r_grid_cm.shape != grid_cache["r_grid"].shape or not np.allclose(
            r_grid_cm,
            grid_cache["r_grid"],
        ):
            grid_cache = None
    return DustRingModel(
        alpha,
        vf,
        float(model_cfg["sigma_g"]),
        eps * eps_scale,
        float(model_cfg["pressure_width_au"]) * c.au,
        temp,
        R_0=float(model_cfg["ring_center_au"]) * c.au,
        r_grid=np.asarray(r_grid_au, dtype=float) * c.au,
        M_star=float(model_cfg["stellar_mass_msun"]) * c.M_sun,
        rho_s=problem.opacity["rho_s"],
        size_res=int(model_cfg.get("size_res", 140)),
        a_min=float(model_cfg.get("a_min", 1e-5)),
        a_max=float(model_cfg.get("a_max", 100.0)),
        T_slope=float(model_cfg.get("temperature_slope", -0.5)),
        q=float(model_cfg.get("size_distribution_q", -3.5)),
        grid_cache=grid_cache,
    )


def radial_cell_widths(r_cm):
    r_cm = np.asarray(r_cm, dtype=float)
    if r_cm.size < 2:
        return np.zeros_like(r_cm)
    dr = np.empty_like(r_cm)
    dr[1:-1] = 0.5 * (r_cm[2:] - r_cm[:-2])
    dr[0] = r_cm[1] - r_cm[0]
    dr[-1] = r_cm[-1] - r_cm[-2]
    return dr


def dust_mass_earth(model):
    sigma_dust_total = np.asarray(model.SigmaDust, dtype=float).sum(axis=1)
    dr = radial_cell_widths(model.r_grid)
    mass_g = np.sum(2.0 * np.pi * model.r_grid * dr * sigma_dust_total)
    return mass_g / c.M_earth


def tau_abs_profile(model, opacity, wavelength_cm):
    lam = np.asarray(opacity["lam"], dtype=float)
    if wavelength_cm < lam.min() or wavelength_cm > lam.max():
        raise ValueError(
            f"tau wavelength {wavelength_cm:g} cm is outside opacity range "
            f"[{lam.min():g}, {lam.max():g}] cm"
        )

    k_abs_at_lam = np.array(
        [np.interp(wavelength_cm, lam, opacity["k_abs"][ia, :]) for ia in range(len(opacity["a"]))],
        dtype=float,
    )
    k_abs_model = np.interp(model.size, opacity["a"], k_abs_at_lam)
    tau_abs = (model.SigmaDust * k_abs_model[None, :]).sum(axis=1)
    return model.r_grid.copy(), tau_abs


def data_chi2(problem, params, r_grid_au, model):
    if hasattr(problem, "opacity_at_model_size"):
        raw = DustRingModel.scattering_precomputed(
            problem.wavelengths,
            model.SigmaDust,
            model.T,
            problem.inclination_deg,
            problem.opacity_at_model_size["k_abs"],
            problem.opacity_at_model_size["k_sca_eff"],
            scattering_formula=getattr(problem, "scattering_formula", "zhu2019"),
            kataoka_coeffs=getattr(problem, "kataoka_coeffs", None),
        )
    else:
        raw = DustRingModel.scattering(
            problem.wavelengths,
            model.SigmaDust,
            model.T,
            model.size,
            problem.inclination_deg,
            problem.opacity["a"],
            problem.opacity["lam"],
            problem.opacity["k_abs"],
            problem.opacity["k_sca"],
            problem.opacity["g"],
            scattering_formula=getattr(problem, "scattering_formula", "zhu2019"),
            kataoka_coeffs=getattr(problem, "kataoka_coeffs", None),
        )

    cal_factors = np.asarray(params[4 : 4 + problem.n_band], dtype=float)
    chi2 = 0.0
    ndata_raw = 0
    ndata_eff = 0.0
    for iband in range(problem.n_band):
        if hasattr(problem, "fit_projection_matrices"):
            model_at_data = problem.fit_projection_matrices[iband] @ raw[iband]
            radii_au = problem.fit_radii_au[iband]
            obs_intensity = problem.fit_obs_intensity[iband]
            rms_eff = problem.fit_rms_eff[iband]
            oversampling = problem.fit_oversampling[iband]
        elif hasattr(problem, "convolution_matrices"):
            profile = problem.convolution_matrices[iband] @ raw[iband]
            radii_au = problem.fit_radii_au[iband]
            model_at_data = np.interp(radii_au, r_grid_au, profile)
            obs_intensity = problem.fit_obs_intensity[iband]
            rms_eff = problem.fit_rms_eff[iband]
            oversampling = problem.fit_oversampling[iband]
        else:
            beam_au = problem.beams_arcsec[iband] * problem.distance_pc
            profile = convolve_radial_profile(r_grid_au, raw[iband], beam_au)
            idx = problem.fit_indices[iband]
            radii_au = problem.radius_au[iband][idx]
            model_at_data = np.interp(radii_au, r_grid_au, profile)
            obs_intensity = problem.obs_intensity[iband][idx]
            oversampling = problem.oversampling_factors(iband, radii_au)
            rms_eff = problem.obs_rms[iband][idx] * np.sqrt(oversampling)

        model_at_data = cal_factors[iband] * model_at_data
        residual = (obs_intensity - model_at_data) / rms_eff
        chi2 += float(np.sum(residual**2))
        ndata_raw += residual.size
        ndata_eff += float(np.sum(1.0 / oversampling))
    return chi2, ndata_raw, ndata_eff


def reduced_chi2_denominator(problem, ndata_raw, ndata_eff, npar):
    """Return the denominator used for the reported reduced data chi-square.

    By default this reports chi-square per fitted data point, i.e. the mean
    normalized squared residual. This is more stable than a formal reduced
    chi-square when the number of fitted radial-profile points is small.
    Set ``reduced_chi2_dof: raw_dof`` to report the ordinary sampled-data
    reduced chi-square.
    """
    mode = problem.config.get("reduced_chi2_dof", "raw_data")
    if mode == "effective_data":
        denom = float(ndata_eff)
    elif mode == "raw_data":
        denom = float(ndata_raw)
    elif mode == "effective_dof":
        denom = float(ndata_eff) - float(npar)
    elif mode == "raw_dof":
        denom = float(ndata_raw) - float(npar)
    else:
        raise ValueError(
            "reduced_chi2_dof must be one of: "
            "effective_data, raw_data, effective_dof, raw_dof"
        )
    return denom if denom > 0 else np.nan


def evaluate_sample(problem, sample, tau_wavelength_cm=0.13, eps_scale=1.0):
    params = complete_params(problem, sample)
    r_grid_au = getattr(problem, "model_r_grid_au", None)
    if r_grid_au is None:
        r_grid_au = model_r_grid_au(problem.config)
    model = make_model(problem, params, r_grid_au, eps_scale=eps_scale)
    model.run_model()

    chi2, ndata_raw, ndata_eff = data_chi2(problem, params, r_grid_au, model)
    npar = int(problem.config.get("reduced_chi2_n_parameters", len(problem.parameters)))
    red_chi2_denom = reduced_chi2_denominator(problem, ndata_raw, ndata_eff, npar)
    loglike_total = -0.5 * chi2
    if problem.config.get("include_calibration_penalty", True):
        cal_factors = np.asarray(params[4 : 4 + problem.n_band], dtype=float)
        loglike_total += float(
            -0.5 * np.sum(((cal_factors - 1.0) / problem.calerr) ** 2)
        )

    a_frag = float(model.aFrag_0)
    st0 = 0.5 * np.pi * problem.opacity["rho_s"] * a_frag / float(problem.config["model"]["sigma_g"])
    w_eps = float(model.width_eps[0] / c.au) if hasattr(model, "width_eps") else np.nan

    tau_r_cm, tau_abs = tau_abs_profile(model, problem.opacity, tau_wavelength_cm)
    tau_at_r0 = float(
        np.interp(float(problem.config["model"]["ring_center_au"]) * c.au, tau_r_cm, tau_abs)
    )

    return {
        "alpha": float(params[0]),
        "vf": float(params[1]),
        "eps": float(params[2]),
        "T": float(params[3]),
        "a_frag": a_frag,
        "St0": float(st0),
        "M_dust": float(dust_mass_earth(model)),
        "w_eps": w_eps,
        "tau_abs": tau_at_r0,
        "chi2_data": float(chi2),
        "red_chi2_data": float(chi2 / red_chi2_denom) if np.isfinite(red_chi2_denom) else np.nan,
        "ndata_raw": float(ndata_raw),
        "ndata_eff": float(ndata_eff),
        "red_chi2_denom": float(red_chi2_denom),
        "loglike_total": loglike_total,
    }


def evaluate_posterior(
    config_path=None,
    posterior_path=None,
    samples=None,
    tau_wavelength_cm=0.13,
    eps_scale=1.0,
    max_samples=None,
    stride=1,
    progress=True,
    result_dir=None,
):
    """Evaluate derived quantities for posterior samples.

    Parameters
    ----------
    config_path : str or Path, optional
        YAML config used for the fit, or a result directory containing
        ``config.yaml``.
    result_dir : str or Path, optional
        Result directory containing ``config.yaml`` and
        ``chains/equal_weighted_post.txt``. Prefer this in notebooks when each
        fit keeps its config beside the posterior.
    posterior_path : str or Path, optional
        If omitted, uses ``output_dir/chains/equal_weighted_post.txt`` from the
        config.
    samples : array, optional
        Pre-selected posterior samples. If provided, ``posterior_path`` is not
        read.
    tau_wavelength_cm : float
        Wavelength for the absorption optical depth column.
    eps_scale : float
        Optional scale factor for eps when recomputing derived quantities.
        Default is 1.0, matching the sampled model.
    """
    if result_dir is not None:
        config_path = result_dir
    if config_path is None:
        raise ValueError("provide either result_dir or config_path")

    config_path, resolved_posterior_path = resolve_result_inputs(config_path, posterior_path)
    config = load_config(config_path)
    problem = RingFitProblem(config)

    if samples is None:
        samples = load_posterior(resolved_posterior_path)
    samples = np.asarray(samples, dtype=float)[:: int(stride)]
    if max_samples is not None:
        samples = samples[: int(max_samples)]

    rows = []
    for i, sample in enumerate(samples):
        if progress and (i == 0 or (i + 1) % 100 == 0 or i + 1 == len(samples)):
            print(f"derived posterior sample {i + 1}/{len(samples)}")
        rows.append(evaluate_sample(problem, sample, tau_wavelength_cm=tau_wavelength_cm, eps_scale=eps_scale))

    table = {key: np.array([row[key] for row in rows], dtype=float) for key in DERIVED_COLUMNS}
    param_samples = samples[:, : min(samples.shape[1], len(problem.parameters))]
    for iparam, name in enumerate(problem.parameters[: param_samples.shape[1]]):
        if name not in table:
            table[name] = param_samples[:, iparam]
    table["_n_samples"] = len(samples)
    table["_tau_wavelength_cm"] = tau_wavelength_cm
    table["_config_path"] = str(config_path)
    table["_posterior_path"] = str(resolved_posterior_path)
    return table, problem


def _trim_zeros(text):
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _format_fixed(x, decimals):
    return _trim_zeros(f"{x:.{decimals}f}")


def _sig_digits_from_error(err):
    if not np.isfinite(err) or err <= 0:
        return 2
    exp = int(np.floor(np.log10(abs(err))))
    lead = err / (10 ** exp)
    return 2 if lead < 3 else 1


def _decimals_from_error(err, sig_digits=2):
    if not np.isfinite(err) or err <= 0:
        return 1
    exp = int(np.floor(np.log10(abs(err))))
    return max(sig_digits - 1 - exp, 0)


def format_pm_plain(median, err_low, err_high, force_decimals=None):
    if force_decimals is None:
        sig = min(_sig_digits_from_error(err_low), _sig_digits_from_error(err_high))
        decimals = max(_decimals_from_error(err_low, sig), _decimals_from_error(err_high, sig))
    else:
        decimals = int(force_decimals)
    return rf"${_format_fixed(median, decimals)}^{{+{_format_fixed(err_high, decimals)}}}_{{-{_format_fixed(err_low, decimals)}}}$"


def format_pm_sci(median, err_low, err_high, exponent=None):
    if median == 0 or not np.isfinite(median):
        return format_pm_plain(median, err_low, err_high)
    if exponent is None:
        exponent = int(np.floor(np.log10(abs(median))))
    for _ in range(2):
        scale = 10.0**exponent
        m = median / scale
        lo = err_low / scale
        hi = err_high / scale
        sig = min(_sig_digits_from_error(lo), _sig_digits_from_error(hi))
        decimals = max(_decimals_from_error(lo, sig), _decimals_from_error(hi, sig))
        rounded_m = abs(float(f"{m:.{decimals}f}"))
        if rounded_m < 10.0:
            break
        exponent += 1
    return rf"${_format_fixed(m, decimals)}^{{+{_format_fixed(hi, decimals)}}}_{{-{_format_fixed(lo, decimals)}}}\times10^{{{exponent}}}$"


def format_pm_auto(values, style="auto", force_decimals=None, sci_threshold_low=1e-2, sci_threshold_high=1e3):
    median, err_low, err_high = median_pm(values)
    if style == "plain":
        return format_pm_plain(median, err_low, err_high, force_decimals=force_decimals)
    if style == "sci":
        return format_pm_sci(median, err_low, err_high)
    if style != "auto":
        raise ValueError("style must be 'auto', 'plain', or 'sci'")
    x = abs(median)
    if x != 0 and (x < sci_threshold_low or x >= sci_threshold_high):
        return format_pm_sci(median, err_low, err_high)
    return format_pm_plain(median, err_low, err_high, force_decimals=force_decimals)


def format_scalar(x, decimals=1):
    return _trim_zeros(f"{float(x):.{decimals}f}")


def format_weight(weight, decimals=2):
    return _trim_zeros(f"{float(weight):.{decimals}f}")


def format_summary_table(table, columns=None):
    if columns is None:
        cal_columns = sorted(
            [name for name in table if isinstance(name, str) and name.startswith("c_lam")],
            key=lambda name: int(name[5:]) if name[5:].isdigit() else name,
        )
        columns = [
            "alpha",
            "vf",
            "T",
            "eps",
            "a_frag",
            "tau_abs",
            "M_dust",
            "w_eps",
            "red_chi2_data",
            *cal_columns,
        ]
    lines = []
    for col in columns:
        if col not in table:
            continue
        q16, q50, q84 = q1684(table[col])
        lines.append(f"{col:>14s}: {q16:.6e}, {q50:.6e}, {q84:.6e}")
    return "\n".join(lines) + "\n"


def summarize_table(table, columns=None, output=None):
    text = format_summary_table(table, columns=columns)
    print(text, end="")
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return text


def make_latex_row(
    table,
    source,
    mode,
    mode_weight,
    source_radius=None,
    include_tau=True,
    chi2_agg="min",
):
    """Return a paper-table LaTeX row from a derived posterior table."""
    if source_radius is not None:
        src = f"{source}({format_scalar(source_radius, 1)})"
    else:
        src = source
    mode_str = f"{mode}(${format_weight(mode_weight, 2)}$)"

    fields = [
        src,
        mode_str,
        format_pm_auto(table["alpha"], style="sci"),
        format_pm_auto(table["vf"], style="plain"),
        format_pm_auto(table["T"], style="plain", force_decimals=1),
        format_pm_auto(table["eps"], style="sci"),
        format_pm_auto(table["a_frag"], style="sci"),
    ]
    if include_tau:
        fields.append(format_pm_auto(table["tau_abs"], style="auto"))
    fields.extend(
        [
            format_pm_auto(table["M_dust"], style="plain"),
            format_pm_auto(table["w_eps"], style="plain", force_decimals=1),
        ]
    )

    if chi2_agg == "min":
        chi2_value = np.nanmin(table["red_chi2_data"])
    elif chi2_agg == "median":
        chi2_value = np.nanmedian(table["red_chi2_data"])
    else:
        raise ValueError("chi2_agg must be 'min' or 'median'")
    fields.append(f"${format_scalar(chi2_value, 1)}$")
    return " &\n".join(fields) + r" \\"


def make_rule(ncol=11):
    return rf"\multicolumn{{{ncol}}}{{c}}{{\rule{{\linewidth}}{{0.4pt}}}} \\"
