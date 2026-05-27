"""Public API for the ``dring`` dust-ring fitting package."""

from .api import (
    __version__,
    compute_profile,
    demo,
    derived_table,
    fit,
    load_config,
    make_model,
    plot_intensity_fit,
    plot_result,
)
from ringfit.model import DustRingModel, convolve_radial_profile

__all__ = [
    "__version__",
    "DustRingModel",
    "RingFitProblem",
    "compute_profile",
    "convolve_radial_profile",
    "demo",
    "derived_table",
    "fit",
    "load_config",
    "make_model",
    "plot_intensity_fit",
    "plot_result",
]


def __getattr__(name):
    if name == "RingFitProblem":
        try:
            from ringfit.fitter import RingFitProblem
        except ImportError as exc:
            raise RuntimeError(
                "RingFitProblem requires optional fitting dependencies. "
                "Install them with: pip install 'dring[fit]'"
            ) from exc
        return RingFitProblem
    raise AttributeError(f"module 'dring' has no attribute {name!r}")
