"""install: put on this machine what this router is made of.

Two things, and they belong to different machines. `llamacpp` is the engine the router
runs, on the machine with the card. `pi` is the agent that calls it, on any machine that
does -- including that one, where somebody works at the card as well as serving from it.

The loop is here and it decides nothing: engine.py installs a release, client.py
installs pi and its extension, and this reads the command line and reports a refusal.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import client, engine, reading, upstream
from .config import ConfigError
from .machine import UnreadableDevice
from .pi import CONFIG
from .refusal import Refusal


def main(argv: Sequence[str] | None = None) -> int:
    given = _arguments(argv if argv is not None else sys.argv[1:])

    try:
        given.act(given)
    except (ConfigError, Refusal, UnreadableDevice,
            upstream.Unavailable) as refusal:
        sys.stdout.flush()
        print(refusal, file=sys.stderr)
        return 1

    return 0


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="install",
        description="Install what this machine is missing, or bring it up to date.")
    what = parser.add_subparsers(required=True)

    llamacpp = what.add_parser(
        "llamacpp", help="the llama.cpp release the router runs",
        description="Install the newest llama.cpp release published for this "
                    "machine's CUDA version, and remove the ones past keeping.")
    llamacpp.add_argument("--settings", type=Path, default=reading.DEFAULT,
                          help="the settings file to read")
    llamacpp.add_argument("--check", action="store_true",
                          help="report what is installed and what is available, and "
                               "download nothing")
    llamacpp.add_argument("--force", action="store_true",
                          help="install even where the newest release is already here, "
                               "after a partial or damaged install")
    llamacpp.set_defaults(act=_llamacpp)

    pi = what.add_parser(
        "pi", help="the pi agent and the context-policy extension",
        description="Install or update the pi agent and the context-policy extension "
                    "on this machine. What points it at a router is cm.pi.")
    pi.add_argument("--config", type=Path, default=Path.home() / CONFIG,
                    help="pi's model configuration")
    pi.set_defaults(act=_pi)

    return parser.parse_args(list(argv))


def _llamacpp(given: argparse.Namespace) -> None:
    engine.install(given.settings, given.check, given.force)


def _pi(given: argparse.Namespace) -> None:
    client.install(given.config)


if __name__ == "__main__":
    sys.exit(main())
