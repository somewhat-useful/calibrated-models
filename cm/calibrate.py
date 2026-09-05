"""calibrate: work out where each model in the settings file sits on this card.

The loop is here and it decides nothing. The core says which configuration to ask the
estimator about, this asks, and the core decides once the answers are in. Every number
that ends up in the preset was computed against the card in this machine.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import (devices, files, invoke, place, proc, reading, releases, render,
               report, workspace)
from .config import Config, ConfigError, Model
from .estimate import parse_requirement
from .facts import parse_facts
from .machine import UnreadableDevice, system_memory
from .name import names
from .place import Limits
from .render import Placed
from .units import Mib

# What the estimator is called. It reads the header of a GGUF and works out what a
# placement would need; it loads nothing and takes about half a second.
ESTIMATOR = "llama-fit-params.exe"


def main(argv: Sequence[str] | None = None) -> int:
    given = _arguments(argv if argv is not None else sys.argv[1:])

    try:
        _calibrate(given.settings)
    except (ConfigError, UnreadableDevice) as refusal:
        print(refusal, file=sys.stderr)
        return 1

    return 0


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="calibrate",
        description="Place every model in the settings file on this machine's card "
                    "and write the router's preset file.")
    parser.add_argument("--settings", type=Path, default=reading.DEFAULT,
                        help="the settings file to read")
    return parser.parse_args(list(argv))


def _calibrate(settings: Path) -> None:
    read = reading.read(settings)
    if not files.exists(read.model_root):
        raise ConfigError(f"model_root does not exist: {read.model_root}")
    if not read.models:
        raise ConfigError(_names_nothing(settings, read))

    estimator = _estimator(workspace.engines())
    machine = devices.probe()
    limits = place.limits_for(machine.card.total, read.reserve,
                              read.min_ctx, read.ample_ctx)

    print(report.opening(machine.card, limits.available, limits.reserve))
    print()

    placed = []
    for model in read.models:
        one = _place(estimator, read, model, limits)
        placed.append(one)
        print("\n".join(report.about(one)))

    memory = system_memory(read.cache_ram, machine,
                           Mib(max((one.resident for one in placed), default=0)))

    preset = settings.parent / read.preset_path
    files.write(preset, render.preset(read, machine, memory, placed))

    print()
    print(report.system(memory))
    print(report.closing(preset, placed))


def _names_nothing(settings: Path, read: Config) -> str:
    """Nothing to place, which is two different situations and two different fixes."""
    if read.withheld:
        return (f"{settings.name} names {len(read.withheld)} model(s) and every one of "
                "them is set aside with hidden = true, so there is nothing to place.")

    return (f"{settings.name} names no model, so there is nothing to place.\n"
            "Write an entry for everything in the library: python -m cm.scan")


def _estimator(engines: Path) -> Path:
    """The newest unpacked release that carries the estimator."""
    for release in releases.releases(files.directories(engines)):
        binary = engines / release.name / ESTIMATOR
        if files.exists(binary):
            return binary

    raise ConfigError(f"No llama.cpp release under {engines} carries {ESTIMATOR}.\n"
                      "Install one: python -m cm.install llamacpp")


def _place(estimator: Path, read: Config, model: Model, limits: Limits) -> Placed:
    """One model, asked about until the core stops asking."""
    if not files.exists(model.path):
        raise ConfigError(f"{model.key}: file not found: {model.path}")

    facts = parse_facts(proc.run(invoke.facts_argv(estimator, model.path)).err)

    answers = {}
    while asking := place.next_questions(facts, model.allowed, limits, answers):
        for question in asking:
            argv = invoke.argv(estimator, model.path, question, read.runtime)
            answers[question] = parse_requirement(proc.run(argv).out)

    chosen = place.settings(facts, model.allowed, limits, answers)
    return Placed(model, names(model.key, chosen), place.resident(chosen, answers))


if __name__ == "__main__":
    sys.exit(main())
