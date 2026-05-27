"""Forward model for a pressure-bump dust ring."""

import numpy as np

from . import constants as c

SIGMA_DUST_FLOOR = 1e-30

try:
    from numba import njit
except Exception:  # pragma: no cover - numba is an optional speed dependency.
    njit = None


def _scattering_rt_numpy(obs_lam, T, theta, kav_abs, kav_sca_eff):
    obs_lam = np.atleast_1d(obs_lam)
    denom = kav_sca_eff + kav_abs
    albedo = np.divide(kav_sca_eff, denom, out=np.zeros_like(kav_sca_eff), where=denom > 0.0)

    tau_d = kav_abs + kav_sca_eff
    mu = np.cos(theta * 2 * np.pi / 360)
    tau = 2 * mu * tau_d / (3 * tau_d + 1)
    intensity = np.zeros((len(obs_lam), len(T)))
    for iwv in range(len(obs_lam)):
        root = np.sqrt(3 * (1 - albedo[iwv]))
        numerator = np.exp(-root * tau[iwv]) + np.exp(root * (tau[iwv] - tau_d[iwv]))
        denominator = np.exp(-root * tau_d[iwv]) * (1 - np.sqrt(1 - albedo[iwv])) + (
            np.sqrt(1 - albedo[iwv]) + 1
        )
        intensity[iwv] = (
            DustRingModel.blackbodylaw(obs_lam[iwv], T)
            * (1 - np.exp(-tau_d[iwv] / mu))
            * (1 - albedo[iwv] * (numerator / denominator))
            * 1e23
        )
    return intensity


if njit is not None:

    @njit(cache=True)
    def _scattering_rt_numba(obs_lam, T, theta, kav_abs, kav_sca_eff):
        n_lam = obs_lam.size
        n_r = T.size
        intensity = np.zeros((n_lam, n_r))
        mu = np.cos(theta * 2.0 * np.pi / 360.0)
        clight = 29979245800.0
        kb = 1.380649e-16
        hplanck = 6.626176e-27

        for iwv in range(n_lam):
            lam = obs_lam[iwv]
            nu = clight / lam
            planck_prefactor = 2.0 * hplanck * nu**3 / clight**2
            for ir in range(n_r):
                abs_i = kav_abs[iwv, ir]
                sca_i = kav_sca_eff[iwv, ir]
                tau_d = abs_i + sca_i
                if tau_d <= 0.0:
                    continue

                albedo = sca_i / tau_d
                one_minus_albedo = max(1.0 - albedo, 0.0)
                sqrt_one_minus_albedo = np.sqrt(one_minus_albedo)
                root = np.sqrt(3.0 * one_minus_albedo)
                tau = 2.0 * mu * tau_d / (3.0 * tau_d + 1.0)

                numerator = np.exp(-root * tau) + np.exp(root * (tau - tau_d))
                denominator = (
                    np.exp(-root * tau_d) * (1.0 - sqrt_one_minus_albedo)
                    + sqrt_one_minus_albedo
                    + 1.0
                )
                if T[ir] <= 0.0:
                    continue
                blackbody_denom = np.exp(hplanck * nu / (kb * T[ir])) - 1.0
                if blackbody_denom <= 0.0:
                    continue
                blackbody = planck_prefactor / blackbody_denom
                intensity[iwv, ir] = (
                    blackbody
                    * (1.0 - np.exp(-tau_d / mu))
                    * (1.0 - albedo * (numerator / denominator))
                    * 1e23
                )
        return intensity

else:
    _scattering_rt_numba = None


class DustRingModel:
    def __init__(
        self,
        alpha_0,
        vf_0,
        Sg_0,
        eps_0,
        W_p,
        T_0,
        xi=11 / 6,
        R_0=100 * c.au,
        M_star=1.9 * c.M_sun,
        r_grid=None,
        a_min=1e-5,
        a_max=100,
        rho_s=1.67,
        size_res=140,
        T_slope=-0.5,
        q=-3.5,
        grid_cache=None,
    ):
        self.M_star = M_star
        self.R_0 = R_0
        self.W_p = W_p
        self.Sg_0 = Sg_0
        self.eps_0 = eps_0
        self.T_0 = T_0
        self.alpha_0 = alpha_0
        self.vf_0 = vf_0
        self.xi = xi
        self.r_grid = np.linspace(90 * c.au, 110 * c.au, 300) if r_grid is None else r_grid
        self.size_res = size_res
        self.log_amin = np.log10(a_min)
        self.log_amax = np.log10(a_max)
        self.q = q
        self.y_a = 1.6
        self.gamma_0 = 1.4
        self.mu = 2.3 * c.m_p
        self.rho_s = rho_s
        self.T_slope = T_slope
        if grid_cache is None:
            grid_cache = self.build_grid_cache(
                self.r_grid,
                a_min,
                a_max,
                self.rho_s,
                self.size_res,
                self.R_0,
                self.M_star,
                self.W_p,
            )
        self._load_grid_cache(grid_cache)

    @staticmethod
    def build_grid_cache(r_grid, a_min, a_max, rho_s, size_res, R_0, M_star, W_p):
        r_grid = np.asarray(r_grid, dtype=float)
        size = np.logspace(np.log10(a_min), np.log10(a_max), int(size_res))
        dlna = np.diff(np.log(size))

        m = rho_s * 4 * np.pi * size**3 / 3
        A = np.mean(m[1:] / m[:-1])
        r_over_R0 = r_grid / R_0
        Ok_0 = np.sqrt(c.G * M_star / R_0**3)
        return {
            "r_grid": r_grid,
            "size": size,
            "size_res": int(size_res),
            "log_amin": np.log10(a_min),
            "log_amax": np.log10(a_max),
            "dlna": np.average(dlna),
            "dlnm": 2 * (A - 1) / (A + 1),
            "r_over_R0": r_over_R0,
            "Ok_0": Ok_0,
            "Ok_base": Ok_0 * r_over_R0 ** (-1.5),
            "pressure_shape": np.exp(-0.5 * ((r_grid - R_0) / W_p) ** 2),
            "iR_0": int(np.argmin(np.abs(r_grid - R_0))),
        }

    def _load_grid_cache(self, grid_cache):
        self.r_grid = grid_cache["r_grid"]
        self.size = grid_cache["size"]
        self.size_res = grid_cache["size_res"]
        self.log_amin = grid_cache["log_amin"]
        self.log_amax = grid_cache["log_amax"]
        self.dlna = grid_cache["dlna"]
        self.dlnm = grid_cache["dlnm"]
        self.r_over_R0 = grid_cache["r_over_R0"]
        self.Ok_0 = grid_cache["Ok_0"]
        self.Ok_base = grid_cache["Ok_base"]
        self.pressure_shape = grid_cache["pressure_shape"]
        self.iR_0 = grid_cache["iR_0"]

    def initial_gas(self):
        size = self.size
        r_grid = self.r_grid
        Ok = self.Ok_base

        T = self.T_0 * self.r_over_R0 ** self.T_slope
        cs = np.sqrt(self.gamma_0 * c.k_B * T / self.mu)
        cs_0 = np.sqrt(self.gamma_0 * c.k_B * self.T_0 / self.mu)

        cs_iso = np.sqrt(c.k_B * T / self.mu)
        Hp = cs_iso / Ok
        Hp_0 = np.sqrt(c.k_B * self.T_0 / self.mu) / self.Ok_0
        rho_0 = self.Sg_0 / (Hp_0 * np.sqrt(2 * np.pi))
        P_0 = rho_0 * cs_0**2 / self.gamma_0
        P = P_0 * self.pressure_shape
        Sg = np.sqrt(2 * np.pi) * Hp * P * self.gamma_0 / cs**2

        self.alpha = np.full_like(r_grid, self.alpha_0)
        self.v_frag = np.full_like(r_grid, self.vf_0)
        self.St = (np.pi / 2) * self.rho_s * size[None, :] / Sg[:, None]

        a0 = 2.5e-6
        St0 = (np.pi / 2) * self.rho_s * a0 / self.Sg_0
        Re0 = self.alpha_0 * self.Sg_0 * c.sigma_H2 / (2 * self.mu)
        Re014 = Re0**0.25
        u_gas0 = cs_0 * np.sqrt(1.5 * self.alpha_0)
        St12_0 = (1.0 / self.y_a) * Re0 ** (-0.5)
        a12_0 = St12_0 * 2 * self.Sg_0 / (np.pi * self.rho_s)

        St_i = (np.pi / 2) * self.rho_s * size / self.Sg_0
        delta_u = np.zeros_like(size)
        m1 = size < a12_0
        m3 = size >= 5 * a12_0
        m2 = ~(m1 | m3)
        St_diff = np.maximum(St_i - St0, 0.0)
        delta_u[m1] = u_gas0 * Re014 * St_diff[m1]
        eps = (size - a12_0) / (4 * a12_0)
        delta_u[m2] = u_gas0 * (
            (1 - eps[m2]) * Re014 * St_diff[m2] + eps[m2] * np.sqrt(3 * St_i[m2])
        )
        delta_u[m3] = u_gas0 * np.sqrt(3 * St_i[m3])
        delta_u[~m1] = np.sqrt(2 / 3) * delta_u[~m1]

        valid = delta_u < self.vf_0
        self.StFrag_0 = np.max(St_i[valid]) if np.any(valid) else np.nan
        self.aFrag_0 = self.StFrag_0 * 2 * self.Sg_0 / (np.pi * self.rho_s)
        return P, size, Ok, T, cs, Hp, Sg

    def find_ribc(self):
        return self.iR_0

    def width_self(self):
        we = 1.2 * np.sqrt(3 * self.alpha[self.iR_0] / self.StFrag_0) * self.W_p
        self.we = we
        self.width_eps = np.zeros_like(self.size)
        self.i_37 = np.argmin(self.size <= (self.aFrag_0 * 0.37))
        small = self.size <= self.aFrag_0 * 0.37
        self.width_eps[small] = we
        self.width_eps[~small] = we * np.sqrt(
            self.St[self.iR_0, self.i_37] / self.St[self.iR_0, ~small]
        )

    def make_B11_SDmesh(self):
        return (
            self.R0_SD[None, :]
            * self.Sg[:, None]
            * np.exp(-0.5 * ((self.r_grid[:, None] - self.R_0) / self.width_eps[None, :]) ** 2)
            / self.Sg_0
        )

    def run_model(self):
        self.P, self.size, self.Ok, self.T, self.cs, self.Hp, self.Sg = self.initial_gas()
        self.iR_0 = self.find_ribc()

        self.eps_crit_drift = np.nan
        self.eps_crit_diff = np.nan
        self.St_crit = np.nan
        self.width_eps = np.full_like(self.size, np.nan)
        self.R0_SD = np.zeros_like(self.size)

        if (not np.isfinite(self.StFrag_0)) or self.StFrag_0 >= 1:
            self.SigmaDust = np.zeros((len(self.r_grid), len(self.size)))
        elif (not np.isfinite(self.aFrag_0)) or self.aFrag_0 < self.size[0] * 5:
            self.SigmaDust = np.zeros((len(self.r_grid), len(self.size)))
        else:
            self.eps_crit_drift = self.v_frag[self.iR_0] ** 2 / (
                self.Ok[self.iR_0] ** 2 * self.W_p**2 * 6 * self.alpha[self.iR_0]
            )
            self.eps_crit_diff = (
                self.alpha[self.iR_0]
                * self.cs[self.iR_0] ** 2
                / (self.Ok[self.iR_0] ** 2 * self.W_p**2)
            )

            fit = np.where(self.size < self.aFrag_0, self.size ** (self.q + 4), 0.0)
            fit_sum = fit.sum()
            if (not np.isfinite(fit_sum)) or fit_sum <= 0.0:
                self.SigmaDust = np.zeros((len(self.r_grid), len(self.size)))
            else:
                self.R0_SD = self.Sg_0 * self.eps_0 * fit / fit_sum

                if self.eps_0 >= self.eps_crit_drift and self.eps_0 > self.eps_crit_diff:
                    self.width_self()

                elif self.eps_0 <= self.eps_crit_diff:
                    eps_safe = max(self.eps_0, 1e-300)
                    eps_ratio = min(eps_safe / self.eps_crit_diff, 1.0)

                    width_factor = (self.eps_crit_diff / eps_safe) ** 2
                    width_factor = min(width_factor, 1e5)

                    self.we = width_factor * self.W_p
                    self.width_eps = np.full_like(self.size, self.we)

                    a_frag_raw = self.aFrag_0
                    a_frag_floor = 10.0 * self.size[0]
                    drop = (1.0 - eps_ratio) / 0.5
                    drop = min(max(drop, 0.0), 1.0)

                    shrink_power = 0.01
                    keep = 1.0 - drop**shrink_power

                    self.aFrag_eff = a_frag_floor + (a_frag_raw - a_frag_floor) * keep
                    self.aFrag_eff = min(self.aFrag_eff, a_frag_raw)
                    self.aFrag_eff = max(self.aFrag_eff, a_frag_floor)

                    fit = np.where(self.size < self.aFrag_eff, self.size ** (self.q + 4), 0.0)
                    fit_sum = fit.sum()

                    if (not np.isfinite(fit_sum)) or fit_sum <= 0.0:
                        self.SigmaDust = np.zeros((len(self.r_grid), len(self.size)))
                    else:
                        self.R0_SD = self.Sg_0 * self.eps_0 * fit / fit_sum  
                    self.aFrag_0=self.aFrag_eff                   
                else:
                    f_width = 3.0
                    St0 = np.asarray(self.St[self.iR_0], dtype=float)
                    self.St_crit = (
                        2.0
                        * self.eps_0
                        * (self.Ok[self.iR_0] * self.W_p / self.cs[self.iR_0]) ** 2
                    )

                    if (not np.isfinite(self.St_crit)) or self.St_crit <= 0.0:
                        self.width_eps = np.full_like(self.size, 1e5 * self.W_p)
                    else:
                        self.width_eps = np.zeros_like(self.size)
                        small = St0 < self.St_crit / f_width
                        large = ~small

                        self.width_eps[small] = self.W_p * np.sqrt(
                            f_width * self.alpha[self.iR_0] / self.St_crit
                        )

                        St0_safe = np.maximum(St0, 1e-300)
                        self.width_eps[large] = self.W_p * np.sqrt(
                            self.alpha[self.iR_0] / St0_safe[large]
                        )
                        self.we = self.width_eps[0]

                self.SigmaDust = self.make_B11_SDmesh()

        x = np.asarray(self.SigmaDust, dtype=float).copy()
        x = np.nan_to_num(
            x,
            nan=SIGMA_DUST_FLOOR,
            posinf=SIGMA_DUST_FLOOR,
            neginf=SIGMA_DUST_FLOOR,
        )
        mask = x[:, 0] == 0.0
        x[mask, 0] = SIGMA_DUST_FLOOR
        self.SigmaDust = x

    def SDplot(
        self,
        output=None,
        *,
        ax=None,
        figsize=(7.8, 5.8),
        dpi=160,
        title="Best-fit dust surface-density distribution",
        levels=None,
        cmap="binary",
        show_ring_center=True,
        show_fragmentation_size=True,
        close=False,
    ):
        """Plot the dust surface-density distribution from ``SigmaDust``.

        The plotted quantity is ``log10(SigmaDust / dlnm)`` on the model's
        radial grid and grain-size grid. Under MPI, only rank 0 creates or
        writes the figure; other ranks return ``None``.
        """
        if _mpi_rank() != 0:
            return None
        if not hasattr(self, "SigmaDust"):
            raise RuntimeError("run_model() must be called before SDplot()")

        import os
        import tempfile
        from pathlib import Path

        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dring-matplotlib"))

        import matplotlib.pyplot as plt

        plot_quantity = np.asarray(self.SigmaDust, dtype=float).T / self.dlnm
        with np.errstate(divide="ignore", invalid="ignore"):
            log_plot = np.log10(plot_quantity)

        if levels is None:
            finite_log = log_plot[np.isfinite(log_plot)]
            levels = np.linspace(np.nanmax(finite_log) - 7.0, np.nanmax(finite_log), 24) if finite_log.size else 24

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.figure

        cf = ax.contourf(
            self.r_grid / c.au,
            self.size,
            log_plot,
            levels=levels,
            cmap=cmap,
            extend="both",
        )
        if show_ring_center:
            ax.axvline(self.R_0 / c.au, color="tab:red", lw=1.5, ls="--", alpha=0.8)
        if show_fragmentation_size and np.isfinite(getattr(self, "aFrag_0", np.nan)) and self.aFrag_0 > 0:
            ax.axhline(self.aFrag_0, color="tab:blue", lw=1.5, ls="--", alpha=0.8)

        ax.set_yscale("log")
        ax.set_xlabel(r"$r\ \mathrm{[au]}$")
        ax.set_ylabel(r"grain radius $a\ \mathrm{[cm]}$")
        ax.set_title(title)
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(r"$\log_{10}(\Sigma_d / d\ln m)$")
        fig.tight_layout()

        if output is not None:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, bbox_inches="tight")
        if close:
            plt.close(fig)
        return fig, ax

    @staticmethod
    def blackbodylaw(lam, T):
        kb = 1.380649e-16
        clight = 29979245800.0
        mu = clight / lam
        hplanck = 6.626176e-27
        exponential = np.exp((hplanck * mu) / (kb * T)) - 1
        return (2 * hplanck * mu**3) * exponential ** (-1) / (clight**2)

    @staticmethod
    def scattering(obs_lam, SigmaDust, T, size, theta, a, lam, k_abs, k_sca, gsca):
        obs_lam = np.atleast_1d(obs_lam)
        k_a = []
        k_s = []
        g_s = []
        for _lam in obs_lam:
            k_a += [[np.interp(_lam, lam, k_abs[ia, :]) for ia in range(len(a))]]
            k_s += [[np.interp(_lam, lam, k_sca[ia, :]) for ia in range(len(a))]]
            g_s += [[np.interp(_lam, lam, gsca[ia, :]) for ia in range(len(a))]]
        k_a = np.array(k_a)
        k_s = np.array(k_s)
        g_s = np.array(g_s)

        kav_abs = np.zeros((len(obs_lam), len(SigmaDust[:, 0])))
        kav_sca_eff = np.zeros((len(obs_lam), len(SigmaDust[:, 0])))
        albedo = np.zeros((len(obs_lam), len(SigmaDust[:, 0])))
        for iwv in range(len(obs_lam)):
            kav_abs[iwv] = (SigmaDust * np.interp(size, a, k_a[iwv])).sum(1)
            kav_sca_eff[iwv] = (
                SigmaDust * np.interp(size, a, ((1 - g_s) * k_s)[iwv])
            ).sum(1)
            albedo[iwv] = kav_sca_eff[iwv] / (kav_sca_eff[iwv] + kav_abs[iwv])

        tau_d = kav_abs + kav_sca_eff
        mu = np.cos(theta * 2 * np.pi / 360)
        tau = 2 * mu * tau_d / (3 * tau_d + 1)
        intensity = np.zeros((len(obs_lam), len(SigmaDust[:, 0])))
        for iwv in range(len(obs_lam)):
            root = np.sqrt(3 * (1 - albedo[iwv]))
            numerator = np.exp(-root * tau[iwv]) + np.exp(root * (tau[iwv] - tau_d[iwv]))
            denominator = np.exp(-root * tau_d[iwv]) * (1 - np.sqrt(1 - albedo[iwv])) + (
                np.sqrt(1 - albedo[iwv]) + 1
            )
            intensity[iwv] = (
                DustRingModel.blackbodylaw(obs_lam[iwv], T)
                * (1 - np.exp(-tau_d[iwv] / mu))
                * (1 - albedo[iwv] * (numerator / denominator))
                * 1e23
            )
        return intensity

    @staticmethod
    def scattering_precomputed(obs_lam, SigmaDust, T, theta, k_abs_size, k_sca_eff_size):
        obs_lam = np.atleast_1d(obs_lam)
        k_abs_size = np.asarray(k_abs_size, dtype=float)
        k_sca_eff_size = np.asarray(k_sca_eff_size, dtype=float)

        kav_abs = k_abs_size @ SigmaDust.T
        kav_sca_eff = k_sca_eff_size @ SigmaDust.T
        if _scattering_rt_numba is not None:
            return _scattering_rt_numba(obs_lam, np.asarray(T, dtype=float), theta, kav_abs, kav_sca_eff)
        return _scattering_rt_numpy(obs_lam, np.asarray(T, dtype=float), theta, kav_abs, kav_sca_eff)


def convolve_radial_profile(r_au, intensity, beam_fwhm_au, nsigma=5.0):
    r_au = np.asarray(r_au, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    sigma_b = beam_fwhm_au / 2.355

    dr = np.empty_like(r_au)
    dr[1:-1] = 0.5 * (r_au[2:] - r_au[:-2])
    dr[0] = r_au[1] - r_au[0]
    dr[-1] = r_au[-1] - r_au[-2]

    out = np.zeros_like(intensity)
    for i in range(r_au.size):
        d = r_au - r_au[i]
        mask = np.abs(d) <= nsigma * sigma_b
        if not np.any(mask):
            out[i] = intensity[i]
            continue
        weights = np.exp(-0.5 * (d[mask] / sigma_b) ** 2) * dr[mask]
        weights /= weights.sum()
        out[i] = np.sum(intensity[mask] * weights)
    return out


def _mpi_rank():
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
