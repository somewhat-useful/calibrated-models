"""The llama.cpp releases unpacked on this machine: which to run, which to keep.

The release archives extract into a directory named b<number>, optionally followed by
what they were built against: b10448-cuda13.3. Ordering is by that number and not by
when the directory appeared, because an older release extracted this morning is still an
older release.

What runs is the build the settings file records, which is the one install put in place
last. A router and a slave have to run the same build to talk at all, and a build
chosen to match another machine's has to stay chosen when a newer one is unpacked beside
it. Where nothing is recorded -- a machine set up before builds were -- the highest
number present runs, as it always did.

Names in, decisions out. Nothing here opens a directory or removes one.
"""

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

from .upstream import Cuda, directory

_NAME = re.compile(r"^b(\d+)(-.+)?$")


@dataclass(frozen=True)
class Release:
    """An extracted release: what its directory is called, and the build it carries."""

    name: str
    number: int


def releases(names: Sequence[str]) -> tuple[Release, ...]:
    """The releases among these directory names, newest first.

    Anything not named like a release is not one: the directory they are unpacked in
    also holds logs, scripts and whatever else was put there.
    """
    found = [Release(name, int(match.group(1)))
             for name in names
             if (match := _NAME.match(name))]

    return tuple(sorted(found, key=lambda one: one.number, reverse=True))


@dataclass(frozen=True)
class Recorded:
    """The build the settings file records as the one to run."""

    number: int


@dataclass(frozen=True)
class Unrecorded:
    """No build recorded: the newest one unpacked here runs."""


# Which build a program is run from.
Running = Recorded | Unrecorded


def to_run(installed: Sequence[Release], running: Running) -> tuple[Release, ...]:
    """The releases a program is looked for in, in order; the first carrying it runs.

    Recorded, that build and no other: a program taken from another build would be a
    machine quietly running something other than it was set to, and a router and a
    slave that no longer understand each other.
    """
    match running:
        case Recorded(number):
            return tuple(one for one in installed if one.number == number)
        case Unrecorded():
            return tuple(installed)


def unfound(root: Path, program: str, running: Running) -> str:
    """Why there is nothing to run, and the command that puts it there."""
    match running:
        case Recorded(number):
            return (f"The settings file records build {number} as the one to run, and no "
                    f"release of it under {root} carries {program}.\n"
                    f"Install it: python -m cm.install llamacpp --build {number}")
        case Unrecorded():
            return (f"No llama.cpp release under {root} carries {program}.\n"
                    "Install one: python -m cm.install llamacpp")


@dataclass(frozen=True)
class Pruned:
    """What to remove of what is unpacked here, and what was left alone.

    Spared means a server is running from it. Removing it would pull the executable out
    from under a router that is serving requests, and the next model it loads is read
    from those files.
    """

    remove: tuple[Release, ...]
    spared: tuple[Release, ...]


def prune(installed: Sequence[Release], keep: int, running: Collection[str],
          held: Collection[str] = frozenset()) -> Pruned:
    """Which releases are past keeping: everything below the newest `keep` of them.

    At least one is always kept, whatever the settings file says: keeping none would
    delete the release everything is about to run.

    held is never past keeping: the release the settings file records, which is what
    runs, and every one installed by naming its build, which was put there by hand and
    goes the same way.
    """
    older = sorted(installed, key=lambda one: one.number, reverse=True)[max(1, keep):]
    removable = tuple(one for one in older if one.name not in held)

    return Pruned(remove=tuple(one for one in removable if one.name not in running),
                  spared=tuple(one for one in removable if one.name in running))


def built_against(installed: Sequence[Release], cuda: Cuda) -> tuple[Release, ...]:
    """The releases here built against this CUDA version, newest first.

    The runtime is carried over only from one of them. Its libraries are named for the
    major version alone -- cudart64_13.dll under 13.3 and 13.4 both -- so a runtime taken
    from a release of another version would be copied in without a word and run under a
    build it was not shipped with.
    """
    return tuple(one for one in installed if one.name == directory(one.number, cuda))


def missing(present: Collection[str], previous: Collection[str]) -> tuple[str, ...]:
    """The previous release's libraries that this one did not bring its own copy of.

    By name rather than against a list of the CUDA runtime's files written down here: a
    library renamed or added between releases is carried across all the same, and there
    is nothing to revisit when the runtime changes.
    """
    have = {name.lower() for name in present}

    return tuple(sorted(name for name in previous if name.lower() not in have))


def verifies(version: str, build: int) -> bool:
    """Whether the server that printed this is the build that was just unpacked.

    What a half-extracted archive or a download of the wrong release fails, and the
    reason nothing is moved into place before it is asked.

    Bounded on both sides: build numbers run consecutively, so build 1044 reads as the
    opening of build 10448 and a substring would call the wrong one verified.
    """
    return re.search(rf"\bbuild {build}\b", version) is not None
