"""What the router can serve, read back out of the preset file.

`vram` answers one question -- what has to be closed for this model to load -- and it
cannot answer it without knowing what each profile holds. That figure is not in the
settings file a person writes: it was computed against this card by calibrate, at the
same time as the window and the offload. calibrate writes it into the preset it
generates, as a comment the router ignores, and this reads it back.

A profile across several devices says what it holds on each of them. `vram` watches the
card a desktop is drawn on, so what is read is what a profile holds on that card.

Parsing inward is fallible. A preset written by hand, or one older than the comment, has
no figure to read, and every way that can happen ends here as one line saying so.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .config import ConfigError
from .render import REQUIRED
from .units import Mib

# The shared block. It says what every model runs with rather than naming a model, so it
# is nothing a person can ask the router to load.
SHARED = "*"

# A requirement that names its devices: an amount on each, separated by commas.
_ON = re.compile(r"(\d+) MiB on (\S+?)(?=,|$)")


@dataclass(frozen=True)
class Loadable:
    """One profile the router serves, and what it holds on the card watched once it is
    loaded."""

    name: str
    needs: Mib


def parse(text: str, device: str) -> tuple[Loadable, ...]:
    """Every profile the preset offers, in the order the file lists them, with what each
    holds on the device llama.cpp calls by this name."""
    profiles = [(name, lines) for name, lines in _blocks(text) if name != SHARED]
    if not profiles:
        raise ConfigError("the preset offers no models; run calibrate")

    return tuple(Loadable(name, _requirement(name, lines, device))
                 for name, lines in profiles)


def _blocks(text: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The file as sections. Whatever stands before the first header belongs to none."""
    blocks: list[tuple[str, list[str]]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            blocks.append((line[1:-1].strip(), []))
        elif blocks:
            blocks[-1][1].append(line)

    return tuple((name, tuple(lines)) for name, lines in blocks)


def _requirement(name: str, lines: Sequence[str], device: str) -> Mib:
    """What a section says it will hold. A section that says nothing is refused.

    Silence is not zero: a profile that needs nothing would be one that always fits, and
    every comparison this feeds would wave it through.
    """
    stated = [line for line in lines if _states_a_requirement(line)]
    if not stated:
        raise ConfigError(f"{name}: the preset does not say what it needs on the card; "
                          f"run calibrate")

    return _amount(name, stated[0], device)


def _states_a_requirement(line: str) -> bool:
    return _comment(line).startswith(f"{REQUIRED}:")


def _comment(line: str) -> str:
    """What a line says as a comment. A line that is not one says nothing."""
    return line[1:].strip() if line.startswith(";") else ""


def _amount(name: str, line: str, device: str) -> Mib:
    """The figure for this device. A requirement naming devices and not this one is a
    profile that holds nothing on it; one naming none was written for a machine's only
    card."""
    said = _comment(line).removeprefix(f"{REQUIRED}:").strip()

    named = _ON.findall(said)
    if named:
        return Mib(sum(int(amount) for amount, on in named if on == device))

    fields = said.split()
    if not fields or not fields[0].isdigit():
        raise ConfigError(f"{name}: cannot read a requirement out of {line!r}")

    return Mib(int(fields[0]))
