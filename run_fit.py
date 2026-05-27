#!/usr/bin/env python3
"""Backward-compatible wrapper for the new ``dring fit`` command."""

from dring.cli import main as _dring_main


def main():
    import sys

    argv = sys.argv[1:]
    if ("--config" in argv or "-c" in argv) and "fit" not in argv:
        argv = ["fit", *argv]
    return _dring_main(argv)


if __name__ == "__main__":
    main()
