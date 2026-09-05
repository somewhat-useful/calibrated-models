"""Whether pi already has the context-policy extension, and what to do about it.

It is installed the ordinary way, from the repository pi's own installer takes. The one
copy that is not ours is a clone in pi's extensions directory: pi loads that instead of
the package, and whoever put it there did so in order to edit it. Installing over it or
pulling into it is taking over somebody else's working copy, and they can bring it up to
date themselves.

So the disk is looked at first, and only what is not a clone is installed or updated.

Facts in, one decision out. Nothing is opened and nothing is run.
"""

from dataclasses import dataclass
from pathlib import Path

# Where it comes from, in the form pi's own installer takes.
SOURCE = "git:github.com/somewhat-useful/context-policy"

# What pi calls it in its list of packages, and the directory a clone of it sits in.
NAME = "context-policy"


@dataclass(frozen=True)
class Cloned:
    """A checkout with its history behind it: somebody's own copy of the extension."""

    at: Path


@dataclass(frozen=True)
class NotCloned:
    """Nothing pi loads instead of the package: no directory there, or one that is not
    a checkout."""


Checkout = Cloned | NotCloned


@dataclass(frozen=True)
class LeaveAlone:
    """Say what is there and touch nothing: it is being worked on by its owner."""

    at: Path


@dataclass(frozen=True)
class Update:
    """pi has it as a package; pi is what brings it up to date."""

    source: str


@dataclass(frozen=True)
class Install:
    """pi does not have it as a package. Install it the ordinary way."""

    source: str


Decided = LeaveAlone | Update | Install


def loaded_from(extensions: Path) -> Path:
    """Where pi loads this extension from, whether or not anything is there."""
    return extensions / NAME


def decided(found: Checkout, listed: str) -> Decided:
    """What to do about the extension, out of what is on disk and what pi lists.

    `listed` is what pi printed when asked for its packages. A machine where that could
    not be asked passes what it got, which is nothing, and the extension is installed:
    installing what is already installed is what pi's own update does anyway.
    """
    match found:
        case Cloned(at):
            return LeaveAlone(at)
        case NotCloned():
            return Update(SOURCE) if NAME in listed else Install(SOURCE)
