"""pi and the context-policy extension, put on the machine that calls the router.

Touches nothing of the card's: pi is an npm package, its extension is installed by pi,
and both are the client's business.

The loop is here and it decides nothing. extension.py says what to do about the
extension from what is on disk and what pi lists, agent.py says what a models.json pi
has never written holds, and this runs npm and pi and reports what they said.

Every step is one command that can be run again on its own, so a step that fails is
reported with that command and the rest carries on. A package server that was
unreachable should not leave the install half done with nothing saying which half.
"""

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import agent, extension, files, proc
from .extension import Cloned, Install, LeaveAlone, NotCloned, Update
from .proc import Output
from .refusal import Refusal

# pi, as npm knows it. --ignore-scripts is in the line its own README gives: pi needs no
# dependency lifecycle scripts for an ordinary install.
PACKAGE = "@earendil-works/pi-coding-agent"

# Where pi loads extensions from, relative to its configuration.
EXTENSIONS = "extensions"


@dataclass(frozen=True)
class Here:
    """A program this machine can run, and where it is."""

    path: Path


@dataclass(frozen=True)
class Nowhere:
    """A program that is not on the path, and what it is called."""

    name: str


Command = Here | Nowhere


def install(config: Path) -> None:
    """pi and its extension, installed or brought up to date."""
    pi = _current(_command("pi"), _command("npm"))

    _extension(pi, extension.loaded_from(config.parent / EXTENSIONS))
    _models(config)

    print()
    print("Point it at the router: python -m cm.pi <the machine with the card>")


def _current(pi: Command, npm: Command) -> Path:
    """pi, installed or brought up to date, and where it is to be run from."""
    match pi:
        case Here(path):
            print(f"pi is here: {path}")
            print("Updating pi and every package it carries that is not pinned ...")
            _optional("The pi update", (str(path), "update", "--all"))
            return path
        case Nowhere(name):
            return _installed(npm, name)


def _installed(npm: Command, name: str) -> Path:
    """pi, put on this machine by npm, and found afterwards."""
    match npm:
        case Nowhere(_):
            raise Refusal(
                "pi was not found, and neither was npm -- pi is an npm package.\n"
                "Install Node.js first, then run this again:\n"
                "  winget install --id OpenJS.NodeJS.LTS")
        case Here(path):
            print(f"pi is not here. Installing {PACKAGE} with npm; this takes a "
                  "minute ...")
            _optional("The pi install",
                      (str(path), "install", "-g", "--ignore-scripts", PACKAGE))
            return _shim(path, name)


def _shim(npm: Path, name: str) -> Path:
    """Where npm just put pi.

    A shim created a moment ago is not on the path this session inherited, so npm is
    asked where it puts them rather than the path being searched again and believed.
    """
    found = shutil.which(name)
    if found:
        return Path(found)

    where = [one.strip() for one in proc.run((str(npm), "prefix", "-g")).out.splitlines()
             if one.strip()]

    match where:
        case [prefix, *_]:
            for candidate in (Path(prefix) / f"{name}.cmd", Path(prefix) / f"{name}.ps1",
                              Path(prefix) / name):
                if files.exists(candidate):
                    return candidate

    raise Refusal(f"npm has installed {PACKAGE}, but {name} is still not on the path "
                  "and npm did not say where it puts its shims.\n"
                  "Open a new terminal, and run this again.")


def _extension(pi: Path, where: Path) -> None:
    """The extension: installed, updated, or left to whoever cloned it."""
    print()
    print(f"The context-policy extension, from {extension.SOURCE}")

    match extension.decided(_checkout(where), _listed(pi)):
        case LeaveAlone(at):
            print(f"  A git checkout is here: {at}")
            print("  Left alone -- pi loads that instead of the package, and it is "
                  "yours to update.")
        case Update(source):
            _optional("The extension update", (str(pi), "update", source))
        case Install(source):
            _optional("The extension install", (str(pi), "install", source))


def _checkout(where: Path) -> extension.Checkout:
    """Whether somebody's own copy is what pi loads from there."""
    return Cloned(where) if files.exists(where / ".git") else NotCloned()


def _listed(pi: Path) -> str:
    """What pi says it carries. What it could not say is nothing, and reads as nothing."""
    listed = proc.run((str(pi), "list"))
    return listed.out + listed.err


def _models(config: Path) -> None:
    """pi's model list, created empty where pi has never run.

    pi writes it on first run. On a machine where it has been installed and not yet
    started there is nothing to add a provider to, and the empty shape pi documents is
    what makes installing and pointing one errand instead of two.
    """
    print()
    if files.exists(config):
        print(f"pi's model list is at {config}")
        return

    files.ensure(config.parent)
    files.write(config, json.dumps(agent.empty(), indent=2) + "\n")
    print(f"Created an empty {config} for pi to take over.")


def _command(name: str) -> Command:
    """A program this machine can run, or that it cannot."""
    found = shutil.which(name)

    return Here(Path(found)) if found else Nowhere(name)


def _optional(what: str, argv: Sequence[str]) -> Output:
    """One step, with what it printed passed through and a failure reported, not raised.

    Every step here can be run again on its own once whatever stopped it is dealt with,
    and the steps after it are worth doing meanwhile.
    """
    print(f"  {' '.join(argv)}")

    done = proc.run(argv)
    for line in (done.out + done.err).splitlines():
        print(f"    {line}")

    if done.code != 0:
        print(f"  {what} did not finish (exit {done.code}). Run the line above again "
              "on its own once the cause is dealt with.")

    return done
