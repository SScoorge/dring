"""Input-data and opacity loading helpers."""

from pathlib import Path

import numpy as np

from . import constants as c


def infer_project_root(path):
    """Infer the package root from a config path or a path inside the project."""
    path = Path(path).expanduser().resolve()
    start = path if path.is_dir() else path.parent
    for parent in (start, *start.parents):
        if (parent / "ringfit").is_dir() and (parent / "configs").is_dir():
            return parent
    if path.parent.name == "configs":
        return path.parent.parent
    return path.parent


def _remap_project_path(path_value, project_root):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(project_root / path)


def load_config(path):
    import yaml

    path = Path(path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    project_root = infer_project_root(path)

    data_cfg = config.get("data", {})
    for band in data_cfg.get("bands", []):
        if band.get("profile_path") is not None:
            band["profile_path"] = _remap_project_path(band["profile_path"], project_root)

    opacity_cfg = config.get("opacity", {})
    for key in ("size_opac", "lam_opac", "k_abs_opac", "k_sca_opac", "g_sca_opac", "rhos_opac"):
        value = opacity_cfg.get(key)
        if isinstance(value, str):
            opacity_cfg[key] = _remap_project_path(value, project_root)

    if config.get("output_dir") is not None:
        config["output_dir"] = _remap_project_path(config["output_dir"], project_root)

    return config


def _radius_to_au(radius, unit="auto"):
    unit = "auto" if unit is None else str(unit).lower()
    if unit == "au":
        return radius
    if unit == "cm":
        return radius / c.au
    if unit != "auto":
        raise ValueError("radius_unit must be 'auto', 'au', or 'cm'")

    finite = radius[np.isfinite(radius)]
    if finite.size == 0:
        return radius
    if np.nanmedian(np.abs(finite)) > 1e6:
        return radius / c.au
    return radius


def _read_named_npz(path):
    obj = np.load(Path(path), allow_pickle=False)
    return {key: obj[key] for key in obj.files}


def load_band_profile(band, radius_unit="auto"):
    """Load one band's radius, intensity, and rms_error arrays."""
    band_label = band.get("band_name", "band")
    if band.get("profile_path") is not None:
        profile = _read_named_npz(band["profile_path"])
    else:
        profile = band

    missing = [key for key in ("radius", "intensity", "rms_error") if key not in profile]
    if missing:
        raise KeyError(f"{band_label} profile is missing required fields: {missing}")

    radius = _radius_to_au(np.asarray(profile["radius"], dtype=float), radius_unit)
    intensity = np.asarray(profile["intensity"], dtype=float)
    rms_error = np.asarray(profile["rms_error"], dtype=float)

    if radius.ndim != 1:
        raise ValueError(f"{band_label} radius must be a 1D array, got shape {radius.shape}")
    if intensity.ndim != 1:
        raise ValueError(f"{band_label} intensity must be a 1D array, got shape {intensity.shape}")
    if rms_error.ndim != 1:
        raise ValueError(f"{band_label} rms_error must be a 1D array, got shape {rms_error.shape}")
    if not (radius.size == intensity.size == rms_error.size):
        raise ValueError(
            f"{band_label} radius, intensity, and rms_error must have the same length; "
            f"got radius={radius.size}, intensity={intensity.size}, rms_error={rms_error.size}"
        )
    return radius, intensity, rms_error


def load_observation_bands(config):
    """Load per-band observation data from the public ``data.bands`` schema."""
    data_cfg = config["data"]
    bands = data_cfg.get("bands")
    if not bands:
        raise KeyError("data.bands must contain at least one wavelength band")

    radius_unit = data_cfg.get("radius_unit", "auto")
    wavelengths = []
    band_names = []
    angular_resolution = []
    calerr = []
    radius = []
    intensity = []
    rms_error = []

    for i, band in enumerate(bands):
        missing = [
            key
            for key in ("wavelength_cm", "angular_resolution_arcsec", "cal_error")
            if key not in band
        ]
        if missing:
            raise KeyError(f"data.bands[{i}] is missing required fields: {missing}")

        r_i, intensity_i, rms_i = load_band_profile(band, radius_unit=radius_unit)
        wavelengths.append(float(band["wavelength_cm"]))
        band_names.append(str(band.get("band_name", f"band{i + 1}")))
        angular_resolution.append(float(band["angular_resolution_arcsec"]))
        calerr.append(float(band["cal_error"]))
        radius.append(r_i)
        intensity.append(intensity_i)
        rms_error.append(rms_i)

    return {
        "wavelengths": np.asarray(wavelengths, dtype=float),
        "band_names": band_names,
        "angular_resolution_arcsec": np.asarray(angular_resolution, dtype=float),
        "calerr": np.asarray(calerr, dtype=float),
        "radius_au": radius,
        "intensity": intensity,
        "rms_error": rms_error,
    }


def normalize_opacity_data(opacity):
    """Normalize public opacity field names to the internal opacity schema."""
    if isinstance(opacity, np.lib.npyio.NpzFile):
        opacity = {key: opacity[key] for key in opacity.files}
    if not isinstance(opacity, dict):
        raise TypeError("opacity data must be a dict or npz file")

    required = ("size_opac", "lam_opac", "k_abs_opac", "k_sca_opac", "g_sca_opac", "rhos_opac")
    missing = [key for key in required if key not in opacity]
    if missing:
        raise KeyError(f"opacity data is missing required fields: {missing}")

    out = {
        "a": np.asarray(_load_array_value(opacity["size_opac"]), dtype=float),
        "lam": np.asarray(_load_array_value(opacity["lam_opac"]), dtype=float),
        "k_abs": np.asarray(_load_array_value(opacity["k_abs_opac"]), dtype=float),
        "k_sca": np.asarray(_load_array_value(opacity["k_sca_opac"]), dtype=float),
        "g": np.asarray(_load_array_value(opacity["g_sca_opac"]), dtype=float),
        "rho_s": float(np.asarray(_load_array_value(opacity["rhos_opac"]), dtype=float).reshape(-1)[0]),
    }
    if out["a"].ndim != 1:
        raise ValueError(f"size_opac must be a 1D array, got shape {out['a'].shape}")
    if out["lam"].ndim != 1:
        raise ValueError(f"lam_opac must be a 1D array, got shape {out['lam'].shape}")
    expected_shape = (out["a"].size, out["lam"].size)
    if out["k_abs"].shape != expected_shape:
        raise ValueError(
            "k_abs_opac must have shape (len(size_opac), len(lam_opac)); "
            f"expected {expected_shape}, got {out['k_abs'].shape}"
        )
    if out["k_sca"].shape != out["k_abs"].shape:
        raise ValueError(
            "k_sca_opac must have the same shape as k_abs_opac; "
            f"expected {out['k_abs'].shape}, got {out['k_sca'].shape}"
        )
    if out["g"].shape != out["k_abs"].shape:
        raise ValueError(
            "g_sca_opac must have the same shape as k_abs_opac; "
            f"expected {out['k_abs'].shape}, got {out['g'].shape}"
        )
    return out


def _load_array_value(value):
    if isinstance(value, (str, Path)):
        return np.load(Path(value), allow_pickle=False)
    return value


def fit_radii_by_band(config, n_band):
    if "fit_radii_by_band_au" in config and config["fit_radii_by_band_au"] is not None:
        values = config["fit_radii_by_band_au"]
        if isinstance(values, dict):
            radii = []
            for i in range(n_band):
                key = str(i)
                if key not in values and i in values:
                    key = i
                radii.append(np.asarray(values[key], dtype=float))
            return radii
        if len(values) != n_band:
            raise ValueError("fit_radii_by_band_au must have one entry per wavelength band")
        return [np.asarray(v, dtype=float) for v in values]

    radii = np.asarray(config["fit_radii_au"], dtype=float)
    return [radii.copy() for _ in range(n_band)]
