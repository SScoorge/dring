"""Interactive dust surface-density explorer for Streamlit.

Run locally from the repository root with:

    streamlit run examples/interactive_model.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dring
from ringfit import constants as c


st.set_page_config(page_title="dring model explorer", layout="wide")


@st.cache_data(show_spinner=False)
def compute_sigma_dust(
    log_alpha: float,
    log_vfrag: float,
    temperature: float,
    log_eps: float,
    sigma_g: float,
    pressure_width_au: float,
    ring_center_au: float,
    stellar_mass_msun: float,
    r_min_au: float,
    r_max_au: float,
    n_r: int,
    size_res: int,
):
    r_grid_au = np.linspace(float(r_min_au), float(r_max_au), int(n_r))
    model = dring.make_model(
        alpha=10.0**float(log_alpha),
        vf=10.0**float(log_vfrag),
        sigma_g=float(sigma_g),
        eps=10.0**float(log_eps),
        pressure_width_au=float(pressure_width_au),
        temperature=float(temperature),
        ring_center_au=float(ring_center_au),
        stellar_mass_msun=float(stellar_mass_msun),
        r_grid_au=r_grid_au,
        size_res=int(size_res),
    )
    plot_quantity = np.asarray(model.SigmaDust, dtype=float).T / model.dlnm
    with np.errstate(divide="ignore", invalid="ignore"):
        log_sigma = np.log10(plot_quantity)
    eps_crit_drift = float(getattr(model, "eps_crit_drift", np.nan))
    eps_crit_diff = float(getattr(model, "eps_crit_diff", np.nan))
    eps0 = float(10.0**log_eps)
    st0 = 0.5 * np.pi * float(model.rho_s) * float(model.aFrag_0) / float(model.Sg_0)
    if np.isfinite(eps_crit_drift) and eps0 >= eps_crit_drift and eps0 > eps_crit_diff:
        regime = "drift/self-concentrated"
    elif np.isfinite(eps_crit_diff) and eps0 <= eps_crit_diff:
        regime = "diffusion-dominated"
    else:
        regime = "intermediate"

    return {
        "r_au": model.r_grid / c.au,
        "size_cm": model.size,
        "log_sigma": log_sigma,
        "a_frag": float(model.aFrag_0),
        "St0": float(st0),
        "eps_crit_drift": eps_crit_drift,
        "eps_crit_diff": eps_crit_diff,
        "regime": regime,
        "ring_center_au": float(model.R_0 / c.au),
        "alpha": float(10.0**log_alpha),
        "vfrag": float(10.0**log_vfrag),
        "eps": eps0,
        "temperature": float(temperature),
    }


def main() -> None:
    st.title("dring model explorer")
    st.caption("Drag the physical parameters to see how the reconstructed dust surface-density distribution changes.")

    with st.sidebar:
        st.header("Sampled parameters")
        log_alpha = st.slider("log10(alpha)", -6.0, -1.0, -3.0, 0.1)
        log_vfrag = st.slider("log10(vfrag [cm/s])", 0.0, 4.0, 2.0, 0.1)
        temperature = st.slider("T0 [K]", 5.0, 50.0, 20.0, 0.5)
        log_eps = st.slider("log10(eps0)", -4.0, 0.0, -2.0, 0.1)

        st.header("Fixed ring setup")
        ring_center_au = st.slider("ring center [au]", 20.0, 160.0, 67.0, 1.0)
        pressure_width_au = st.slider("pressure width [au]", 3.0, 40.0, 14.4, 0.5)
        sigma_g = st.slider("gas surface density [g/cm2]", 0.5, 80.0, 10.5, 0.5)
        stellar_mass_msun = st.slider("stellar mass [Msun]", 0.1, 3.0, 1.9, 0.1)

        st.header("Grid")
        r_min_au = st.number_input("r min [au]", value=25.0, min_value=0.1, step=1.0)
        r_max_au = st.number_input("r max [au]", value=110.0, min_value=1.0, step=1.0)
        n_r = st.slider("radial cells", 80, 500, 240, 20)
        size_res = st.slider("grain-size cells", 50, 220, 120, 10)

    if r_max_au <= r_min_au:
        st.error("r max must be larger than r min.")
        return

    data = compute_sigma_dust(
        log_alpha,
        log_vfrag,
        temperature,
        log_eps,
        sigma_g,
        pressure_width_au,
        ring_center_au,
        stellar_mass_msun,
        r_min_au,
        r_max_au,
        n_r,
        size_res,
    )

    col_plot, col_values = st.columns([3, 1])

    with col_plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=150)
        finite = data["log_sigma"][np.isfinite(data["log_sigma"])]
        if finite.size:
            levels = np.linspace(np.nanmax(finite) - 7.0, np.nanmax(finite), 24)
        else:
            levels = 24

        cf = ax.contourf(
            data["r_au"],
            data["size_cm"],
            data["log_sigma"],
            levels=levels,
            cmap="binary",
            extend="both",
        )
        ax.axvline(data["ring_center_au"], color="tab:red", lw=1.5, ls="--", alpha=0.85)
        if np.isfinite(data["a_frag"]) and data["a_frag"] > 0:
            ax.axhline(data["a_frag"], color="tab:blue", lw=1.5, ls="--", alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel(r"$r\ [\mathrm{au}]$")
        ax.set_ylabel(r"grain radius $a\ [\mathrm{cm}]$")
        ax.set_title("Dust surface-density distribution")
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(r"$\log_{10}(\Sigma_d / d\ln m)$")
        fig.tight_layout()
        st.pyplot(fig)

    with col_values:
        st.subheader("Current values")
        st.metric("alpha", f"{data['alpha']:.3e}")
        st.metric("vfrag", f"{data['vfrag']:.3g} cm/s")
        st.metric("T0", f"{data['temperature']:.1f} K")
        st.metric("eps0", f"{data['eps']:.3e}")
        st.metric("eps_drift", f"{data['eps_crit_drift']:.3e}")
        st.metric("eps_diff", f"{data['eps_crit_diff']:.3e}")
        st.metric("a_frag", f"{data['a_frag']:.3e} cm")
        st.metric("St0", f"{data['St0']:.3e}")
        st.caption(f"Regime: {data['regime']}")


if __name__ == "__main__":
    main()
