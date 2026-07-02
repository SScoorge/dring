from copy import deepcopy
from importlib import resources
from pathlib import Path

import numpy as np

from dring import load_config
from dring.derived import data_chi2
from ringfit.fitter import RingFitProblem


def demo_config_with_formula(formula):
    config_path = resources.files("dring").joinpath("examples/demo.yaml")
    cfg = load_config(config_path)
    example_dir = Path(str(resources.files("dring").joinpath("examples")))
    for band in cfg["data"]["bands"]:
        band["profile_path"] = str(example_dir / Path(band["profile_path"]).name)
    for key, value in cfg["opacity"].items():
        path_value = Path(value)
        cfg["opacity"][key] = str(example_dir / path_value.parent.name / path_value.name)
    cfg["scattering_formula"] = formula
    cfg["model"]["size_res"] = 60
    cfg["model"]["r_grid_n"] = 80
    return cfg


def test_model_profiles_follow_scattering_formula_switch():
    problems = {
        formula: RingFitProblem(demo_config_with_formula(formula))
        for formula in ("zhu2019", "kataoka2026")
    }
    theta = problems["zhu2019"].prior_transform(np.full(len(problems["zhu2019"].parameters), 0.5))

    r_zhu, profile_zhu = problems["zhu2019"].model_profiles(theta)
    r_kat, profile_kat = problems["kataoka2026"].model_profiles(theta)

    assert problems["zhu2019"].config["scattering_formula"] == "zhu2019"
    assert problems["kataoka2026"].config["scattering_formula"] == "kataoka2026"
    assert np.allclose(r_zhu, r_kat)
    assert profile_zhu.shape == profile_kat.shape
    assert np.all(np.isfinite(profile_zhu))
    assert np.all(np.isfinite(profile_kat))
    assert not np.allclose(profile_zhu, profile_kat)


def test_derived_chi2_uses_problem_scattering_formula():
    cfg_zhu = demo_config_with_formula("zhu2019")
    cfg_kat = deepcopy(cfg_zhu)
    cfg_kat["scattering_formula"] = "kataoka2026"
    problem_zhu = RingFitProblem(cfg_zhu)
    problem_kat = RingFitProblem(cfg_kat)
    theta = problem_zhu.prior_transform(np.full(len(problem_zhu.parameters), 0.5))

    model_zhu = problem_zhu.make_ring_model(theta)
    model_kat = problem_kat.make_ring_model(theta)
    chi_zhu, _, _ = data_chi2(problem_zhu, theta, problem_zhu.model_r_grid_au, model_zhu)
    chi_kat, _, _ = data_chi2(problem_kat, theta, problem_kat.model_r_grid_au, model_kat)

    assert np.isfinite(chi_zhu)
    assert np.isfinite(chi_kat)
    assert chi_zhu != chi_kat
