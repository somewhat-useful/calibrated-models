"""The llama.cpp releases unpacked on this machine: which to run, which to keep.

The release archives extract into a directory named b<number>, optionally followed by
what they were built against: b10448-cuda13.3. Ordering is by that number and not by
when the directory appeared, because an older release extracted this morning is still an
older release. The highest number present is therefore what everything runs, which is
what makes unpacking a release enough to make it current.

Names in, decisions out. Nothing here opens a directory or removes one.
"""

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass

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


def current(installed: Sequence[Release], build: int) -> bool:
    """Whether this build is already here, or something newer is.

    Newer counts: what is unpacked here is what runs, so a machine holding a later
    release than the one just published has nothing to install.
    """
    return any(one.number >= build for one in installed)


@dataclass(frozen=True)
class Pruned:
    """What to remove of what is unpacked here, and what was left alone.

    Spared means a server is running from it. Removing it would pull the executable out
    from under a router that is serving requests, and the next model it loads is read
    from those files.
    """

    remove: tuple[Release, ...]
    spared: tuple[Release, ...]


def prune(installed: Sequence[Release], keep: int, running: Collection[str]) -> Pruned:
    """Which releases are past keeping: everything below the newest `keep` of them.

    At least one is always kept, whatever the settings file says: keeping none would
    delete the release everything is about to run.
    """
    older = sorted(installed, key=lambda one: one.number, reverse=True)[max(1, keep):]

    return Pruned(remove=tuple(one for one in older if one.name not in running),
                  spared=tuple(one for one in older if one.name in running))


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
