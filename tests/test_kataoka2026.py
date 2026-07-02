import numpy as np

from ringfit.model import (
    KATAOKA2026_TAU_SWITCH,
    _scattering_rt_kataoka2026,
    _scattering_rt_numpy,
    intensity_kataoka2026_lambda,
    interpolate_kataoka2026_coeffs,
    load_kataoka2026_coeffs,
)


def dummy_coeffs():
    """Small non-physical table for formula tests only."""
    omega_grid = np.array([0.0, 1.0])
    mu_grid = np.array([0.5, 1.0])
    shape = (omega_grid.size, mu_grid.size)
    return {
        "omega_grid": omega_grid,
        "mu_grid": mu_grid,
        "A_I": np.full(shape, 0.8),
        "B_I": np.full(shape, 1.1),
        "I_conv": np.full(shape, 0.9),
        "omega_I": np.zeros(shape),
    }


def test_kataoka2026_shape_and_broadcasting():
    out = intensity_kataoka2026_lambda(
        T=np.array([10.0, 20.0, 30.0]),
        tau_max=np.array([1e-3, 0.1, 1.0]),
        omega=0.5,
        mu=0.7,
        lambda_=0.13,
        coeffs=dummy_coeffs(),
    )
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


def test_kataoka2026_continuity_at_switch():
    coeffs = dummy_coeffs()
    eps = 1e-8
    left = intensity_kataoka2026_lambda(20.0, KATAOKA2026_TAU_SWITCH - eps, 0.4, 0.7, 0.13, coeffs)
    right = intensity_kataoka2026_lambda(20.0, KATAOKA2026_TAU_SWITCH + eps, 0.4, 0.7, 0.13, coeffs)
    assert np.allclose(left, right, rtol=1e-5, atol=0.0)


def test_kataoka2026_thin_limit_linear_in_tau_over_mu():
    coeffs = dummy_coeffs()
    tau = np.array([1e-5, 2e-5])
    out = intensity_kataoka2026_lambda(20.0, tau, 0.0, 0.8, 0.13, coeffs)
    assert np.allclose(out[1] / out[0], 2.0, rtol=1e-6)


def test_kataoka2026_lambda_unit_conversion():
    coeffs = dummy_coeffs()
    in_cm = intensity_kataoka2026_lambda(20.0, 0.1, 0.5, 0.7, 0.13, coeffs, lambda_unit="cm")
    in_mm = intensity_kataoka2026_lambda(20.0, 0.1, 0.5, 0.7, 1.3, coeffs, lambda_unit="mm")
    assert np.allclose(in_cm, in_mm)


def test_kataoka2026_coeffs_load_locally_and_interpolate():
    coeffs = load_kataoka2026_coeffs()
    pars = interpolate_kataoka2026_coeffs(omega=0.85, mu=0.7, coeffs=coeffs)
    assert set(pars) == {"A_I", "B_I", "I_conv", "omega_I"}
    for value in pars.values():
        assert np.isfinite(value)


def test_kataoka2026_official_coeffs_continuity_at_switch():
    coeffs = load_kataoka2026_coeffs()
    eps = 1e-8
    left = intensity_kataoka2026_lambda(20.0, KATAOKA2026_TAU_SWITCH - eps, 0.8, 0.7, 0.13, coeffs)
    right = intensity_kataoka2026_lambda(20.0, KATAOKA2026_TAU_SWITCH + eps, 0.8, 0.7, 0.13, coeffs)
    assert np.allclose(left, right, rtol=1e-5, atol=0.0)


def test_zhu2019_is_lower_than_kataoka2026_for_scattering_case():
    coeffs = load_kataoka2026_coeffs()
    lam = np.array([0.13])
    temperature = np.array([20.0])
    mu = 0.7
    theta_deg = np.rad2deg(np.arccos(mu))
    tau_max = 1.0
    omega = 0.9
    kav_abs = np.array([[tau_max * (1.0 - omega)]])
    kav_sca_eff = np.array([[tau_max * omega]])

    old_i = _scattering_rt_numpy(lam, temperature, theta_deg, kav_abs, kav_sca_eff)[0, 0]
    new_i = _scattering_rt_kataoka2026(lam, temperature, theta_deg, kav_abs, kav_sca_eff, coeffs)[0, 0]

    ratio = old_i / new_i
    assert 0.8 < ratio < 0.95
