"""Command-line interface for ``dring``."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from .api import __version__, demo, derived_table, fit, load_config, plot_intensity_fit, plot_result


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Show defaults while preserving example formatting."""

    def _get_help_string(self, action):
        help_text = action.help or ""
        default = action.default
        if (
            "%(default)" not in help_text
            and default is not None
            and default is not False
            and default is not argparse.SUPPRESS
            and action.option_strings
        ):
            help_text += " (default: %(default)s)"
        return help_text


def mpi_rank() -> int:
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


def _print_check(problem: RingFitProblem) -> None:
    print("Configuration OK")
    print("parameters:", problem.parameters)
    print("n_band:", problem.n_band)
    print("fit_radii_au:", [r.tolist() for r in problem.fit_radii])
    print(
        "model_r_grid_au:",
        [
            float(problem.model_r_grid_au[0]),
            float(problem.model_r_grid_au[-1]),
            int(problem.model_r_grid_au.size),
        ],
    )


def _config_help_text() -> str:
    return textwrap.dedent(
        """\
        dring YAML config quick reference

        For the full reference, see:
          docs/config_help.md

        Minimal structure:
          scattering_formula: zhu2019
          data:
            bands:
              - wavelength_cm: 0.13
                band_name: band6
                angular_resolution_arcsec: 0.05
                cal_error: 0.10
                profile_path: data/profile_band6.npz
          opacity:
            size_opac: data/default_opacity_dsharp/size_opac.npy
            lam_opac: data/default_opacity_dsharp/lam_opac.npy
            k_abs_opac: data/default_opacity_dsharp/k_abs_opac.npy
            k_sca_opac: data/default_opacity_dsharp/k_sca_opac.npy
            g_sca_opac: data/default_opacity_dsharp/g_sca_opac.npy
            rhos_opac: data/default_opacity_dsharp/rhos_opac.npy
          distance_pc: 101.2
          inclination_deg: 46.0
          fit_radii_au: [63.0, 67.0, 71.0]
          model:
            stellar_mass_msun: 1.9
            ring_center_au: 67.0
            pressure_width_au: 14.4
            sigma_g: 21.0
            size_res: 100
            a_min: 1.0e-5
            a_max: 100.0
            temperature_slope: -0.5
            size_distribution_q: -3.5
          priors:
            alpha: {type: flat, scale: log, min: 1.0e-6, max: 1.0e-1}
            vf: {type: flat, scale: log, min: 1.0, max: 1.0e4}
            eps: {type: flat, scale: log, min: 1.0e-3, max: 1.0}
            T: {type: flat, scale: linear, min: 5.0, max: 30.0}

        Common units:
          wavelength_cm                 [cm]
          angular_resolution_arcsec      beam FWHM/effective resolution [arcsec]
          profile radius                 [au]
          profile intensity              [Jy/sr]
          profile rms_error              [Jy/sr]
          distance_pc                    [pc]
          inclination_deg                [deg]
          fit_radii_au                   [au]
          model.stellar_mass_msun        [Msun]
          model.ring_center_au           [au]
          model.pressure_width_au        [au]
          model.sigma_g                  [g/cm^2]
          model.a_min, model.a_max       [cm]
          priors.vf                      [cm/s]
          priors.T                       [K]

        Notes:
          - data.bands can contain any number of bands.
          - Each profile_path must be an .npz with radius, intensity, rms_error.
          - The radial model grid is automatic unless model.r_grid_au is set.
          - Check a config without sampling:
              dring fit -c configs/HD163296_ring1_eg.yaml --check-config
        """
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dring",
        description=textwrap.dedent(
            """\
            Dust ring modeling, forward-profile generation, and Bayesian fitting.

            The command line focuses on the fitting and analysis layer. The lower
            model layer is available from Python as ``import dring``.
            """
        ),
        epilog=textwrap.dedent(
            """\
            Common workflows:
              dring config-help
              dring demo
              dring fit -c configs/HD163296_ring1.yaml --check-config
              dring fit -c configs/HD163296_ring1.yaml
              mpiexec -n 8 dring fit -c configs/HD163296_ring1.yaml
              python -m dring fit -c configs/HD163296_ring1.yaml

              dring plot -r results/HD163296_ring1
              dring derived -r results/HD163296_ring1
              dring intensity -r results/HD163296_ring1

            Install layers:
              pip install dring
              pip install "dring[fit]"
              pip install "dring[fit,mpi]"

            Config reference:
              dring config-help
              docs/config_help.md
            """
        ),
        formatter_class=HelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"dring {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    sub.add_parser(
        "config-help",
        help="Show the YAML config structure and common units.",
        description="Print a compact YAML config reference with common physical units.",
        formatter_class=HelpFormatter,
    )

    demo_parser = sub.add_parser(
        "demo",
        help="Run the packaged demo check.",
        description="Check that the packaged model/profile example can run.",
        epilog=textwrap.dedent(
            """\
            Examples:
              dring demo
              dring demo --run -o results/demo
            """
        ),
        formatter_class=HelpFormatter,
    )
    demo_parser.add_argument("--run", action="store_true", help="Run UltraNest instead of only checking the demo.")
    demo_parser.add_argument("-o", "--output-dir", help="Override the demo output directory.")

    fit_parser = sub.add_parser(
        "fit",
        help="Fit a YAML config.",
        description=textwrap.dedent(
            """\
            Fit multi-wavelength radial intensity profiles from a YAML config.

            The config controls input bands, opacity arrays, priors, output_dir,
            and analysis products. After sampling, rank 0 writes config.yaml,
            corner.png, derived_summary.txt, and intensity_bestfit.png unless
            disabled in the YAML analysis block.
            """
        ),
        epilog=textwrap.dedent(
            """\
            Examples:
              dring config-help
              dring fit -c configs/HD163296_ring1.yaml --check-config
              dring fit -c configs/HD163296_ring1.yaml
              mpiexec -n 8 dring fit -c configs/HD163296_ring1.yaml
              python -m dring fit -c configs/HD163296_ring1.yaml
            """
        ),
        formatter_class=HelpFormatter,
    )
    fit_parser.add_argument("-c", "--config", required=True, help="Path to a YAML config file.")
    fit_parser.add_argument(
        "--check-config",
        action="store_true",
        help="Load inputs and print inferred parameters without running UltraNest.",
    )

    plot_parser = sub.add_parser(
        "plot",
        help="Plot an UltraNest result directory.",
        description=textwrap.dedent(
            """\
            Make the corner plot from a completed result directory.

            By default, plot settings are read from analysis.corner in the saved
            config.yaml. Command-line columns and limits override that config.
            """
        ),
        epilog=textwrap.dedent(
            """\
            Examples:
              dring plot -r results/HD163296_ring1
              dring plot -r results/HD163296_ring1 --columns 0 1 2 3
              dring plot -r results/HD163296_ring1 --xlim -6 -1 --ylim 0 4
            """
        ),
        formatter_class=HelpFormatter,
    )
    plot_parser.add_argument("-r", "--result-dir", required=True, help="Result directory, e.g. results/hd163296_ring1.")
    plot_parser.add_argument("-o", "--output", help="Output image path.")
    plot_parser.add_argument("--columns", nargs="+", type=int, default=None, help="Posterior columns to plot.")
    plot_parser.add_argument("--xlim", nargs=2, type=float, action="append", metavar=("LO", "HI"))
    plot_parser.add_argument("--ylim", nargs=2, type=float, action="append", metavar=("LO", "HI"))

    derived_parser = sub.add_parser(
        "derived",
        help="Print posterior-derived quantity summaries.",
        description=textwrap.dedent(
            """\
            Recompute derived posterior quantities from an existing result
            directory and write derived_summary.txt.

            The summary includes sampled physical parameters, derived dust
            quantities, fit quality, and calibration factors (c_lam*) at the end.
            """
        ),
        epilog=textwrap.dedent(
            """\
            Examples:
              dring derived -r results/HD163296_ring1
              dring derived -r results/HD163296_ring1 --max-samples 100 --quiet
              dring derived -r results/HD163296_ring1 -o summary.txt
            """
        ),
        formatter_class=HelpFormatter,
    )
    derived_parser.add_argument("-r", "--result-dir", required=True, help="Result directory containing config.yaml.")
    derived_parser.add_argument("--tau-wavelength-cm", type=float, default=None, help="Wavelength used for tau_abs.")
    derived_parser.add_argument("--max-samples", type=int, default=None, help="Maximum posterior samples to evaluate.")
    derived_parser.add_argument("--stride", type=int, default=1, help="Use every Nth posterior sample.")
    derived_parser.add_argument("-o", "--output", help="Output summary text path.")
    derived_parser.add_argument("--quiet", action="store_true", help="Disable progress messages.")

    intensity_parser = sub.add_parser(
        "intensity",
        help="Plot observed intensities and the best-fit model.",
        description=textwrap.dedent(
            """\
            Plot observed radial intensity profiles and the best-fit convolved
            model profile from a completed result directory.

            Defaults are read from analysis.intensity in config.yaml. If no
            limits are provided, x uses the full input radius range and y uses a
            log scale based on the fitted intensity range.
            """
        ),
        epilog=textwrap.dedent(
            """\
            Examples:
              dring intensity -r results/HD163296_ring1
              dring intensity -r results/HD163296_ring1 --xlim 0 200
              dring intensity -r results/HD163296_ring1 -o intensity_check.png
            """
        ),
        formatter_class=HelpFormatter,
    )
    intensity_parser.add_argument("-r", "--result-dir", required=True, help="Result directory containing config.yaml.")
    intensity_parser.add_argument("-o", "--output", help="Output image path.")
    intensity_parser.add_argument("--xlim", nargs=2, type=float, metavar=("LO", "HI"), help="Radius limits in au.")
    intensity_parser.add_argument("--ylim", nargs=2, type=float, metavar=("LO", "HI"), help="Intensity limits.")
    intensity_parser.add_argument("--title", help="Override the plot title.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "config-help":
        print(_config_help_text())
        return 0

    if args.command == "demo":
        result = demo(run=args.run, output_dir=args.output_dir)
        if mpi_rank() == 0:
            if args.run:
                results, problem = result
                print(json.dumps({"parameters": problem.parameters, "logz": results.get("logz")}, indent=2))
            else:
                print("Demo OK")
                print("parameters:", result["parameters"])
                print("profile_shape:", list(result["profiles"].shape))
        return 0

    if args.command == "fit":
        config = load_config(args.config)
        if args.check_config:
            problem = fit(config, check_config=True)
            if mpi_rank() == 0:
                _print_check(problem)
            return 0

        results, problem = fit(config)
        if mpi_rank() == 0:
            summary = {
                "parameters": problem.parameters,
                "logz": results.get("logz"),
                "logzerr": results.get("logzerr"),
            }
            print(json.dumps(summary, indent=2))
        return 0

    if args.command == "plot":
        xlim = {"0": args.xlim[0]} if args.xlim else None
        ylim = {"0": args.ylim[0]} if args.ylim else None
        output = plot_result(Path(args.result_dir), output=args.output, columns=args.columns, xlim=xlim, ylim=ylim)
        if output is not None:
            print(f"wrote {output}")
        return 0

    if args.command == "derived":
        derived_table(
            Path(args.result_dir),
            tau_wavelength_cm=args.tau_wavelength_cm,
            max_samples=args.max_samples,
            stride=args.stride,
            progress=not args.quiet,
            summarize=True,
            output=args.output,
        )
        return 0

    if args.command == "intensity":
        output = plot_intensity_fit(
            Path(args.result_dir),
            output=args.output,
            xlim=args.xlim,
            ylim=args.ylim,
            title=args.title,
        )
        if output is not None:
            print(f"wrote {output}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
