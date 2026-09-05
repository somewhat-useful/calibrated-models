"""The llama.cpp release the router runs: installed here, or brought up to date.

The loop is here and it decides nothing. upstream.py reads what the project published
and says which release can be installed, releases.py says what is here already and what
is past keeping, and this asks, downloads, unpacks and removes.

A release is assembled in a staging directory beside the others and moved into place
only once the server in it says it is the build that was asked for. Nothing part-way
downloaded ever becomes the newest release, which matters because the newest is what
everything runs: unpacking is what makes a release current, so a broken one would make
itself current by being unpacked.
"""

import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from . import (config, devices, files, proc, reading, releases, session, upstream,
               workspace)
from .config import ConfigError
from .lmstudio import Found, Library, Missing
from .refusal import Refusal
from .releases import Release
from .serving import SERVER
from .units import Bytes
from .upstream import Absent, Asset, Latest, Present

# GitHub asks every caller to say what it is, and turns away one that does not.
HEADERS = {"User-Agent": "llamacpp-local-updater"}

# How long to wait for an answer. Not a deadline for a download: it bounds each read,
# and a download that is arriving is answering.
TIMEOUT = 120

_MEGABYTE = 1048576


def install(settings: Path, check: bool, force: bool) -> None:
    """The newest release published for this machine's CUDA version, put here."""
    _settings(settings)

    read = reading.read(settings)
    root = workspace.engines()

    installed = releases.releases(files.directories(root))
    print(f"llama.cpp under {root}, CUDA {read.cuda.version}")
    print("Installed: " + (", ".join(one.name for one in installed) or "none"))

    latest = upstream.latest(upstream.published(_answer()), read.cuda)
    if latest.incomplete:
        print("Skipped (archive not uploaded yet): "
              + ", ".join(f"b{build}" for build in latest.incomplete))
    print(f"Latest:    b{latest.build} (published {latest.when})")

    if releases.current(installed, latest.build) and not force:
        print("Already on the latest build. Nothing to do.")
        return

    target = root / upstream.directory(latest.build, read.cuda)

    if check:
        _would(installed, latest, target)
        return

    files.ensure(root)
    _assemble(root, installed, latest, target)
    _prune(root, read.keep_releases)

    print()
    print(f"Now running: {_server(root)}")
    print("Restart the router to run on it: python -m cm.router start")


def _settings(settings: Path) -> None:
    """The settings file, made from the one that ships where this machine has none yet.

    Copied rather than refused: somebody who has just been handed the scripts should not
    have to write a file by hand to use them. What the copy cannot answer for itself is
    asked for once, here, rather than by every command that reads it afterwards.
    """
    if files.exists(settings):
        return

    template = workspace.template()
    if not files.exists(template):
        raise ConfigError(f"{settings.name} not found at {settings}, and neither is "
                          f"the file it would be copied from: {template}")

    files.write(settings,
                _filled(files.read(template), devices.library(), settings))

    print(f"Copied {template.name} to {settings}")
    print("It names no model yet. Write an entry for each one in the library, and look "
          "the names over: python -m cm.scan")
    print()


def _filled(template: str, library: Library, settings: Path) -> str:
    """The template as this machine needs it before anything can read it.

    One key cannot be defaulted and cannot be read off the machine: where the models
    are. LM Studio answers it where LM Studio is installed, and where it is not, the
    person running this is the only one who knows.
    """
    match library:
        case Found(_):
            return template
        case Missing(looked):
            return config.naming_the_library(template, _library(looked, settings))


def _library(looked: Path, settings: Path) -> Path:
    """Where the models are, asked for.

    Asked once, here, and written into the file: the alternative is a settings file
    that every later command refuses for the same reason, which is a worse way of
    asking the same question.
    """
    if not sys.stdin.isatty():
        raise ConfigError(
            f"There is no LM Studio library at {looked}, and there is nobody to ask: "
            "this is not a terminal.\n"
            f"Copy {workspace.TEMPLATE} to {settings.name} yourself and set "
            "model_root to the directory the models are under.")

    print(f"There is no LM Studio library at {looked}, so nothing on this machine says "
          "where the models are.")

    said = _said("Directory the models are under: ")
    if not said:
        raise ConfigError("Nothing was said, so nothing was written. Run this again, "
                          "or copy the template yourself and set model_root.")

    where = Path(said)
    if not files.exists(where):
        print(f"  {where} is not there yet -- models will report every file missing "
              "until it is.")

    return where


def _said(question: str) -> str:
    """One answer, as a person types it.

    Quotes stripped: a path pasted out of Explorer arrives in them, and a directory
    called "D:\\models" with the quotes in the name is not what anybody meant.
    """
    try:
        return input(question).strip().strip('"')
    except EOFError:
        return ""


def _would(installed: Sequence[Release], latest: Latest, target: Path) -> None:
    """What installing would take, with nothing downloaded and nothing written."""
    print()
    match installed:
        case (newest, *_):
            print(f"Update available: {newest.name} -> b{latest.build}")
        case _:
            print(f"Nothing installed yet: would install b{latest.build}.")

    print(f"Would download {latest.binaries.name} ({_megabytes(latest.binaries.size)})")

    match (installed, latest.runtime):
        case ((), Absent(name)):
            raise Refusal("There would be nothing to carry the CUDA runtime over from, "
                          f"and b{latest.build} carries no {name}.")
        case ((), Present(asset)):
            print(f"Would also download {asset.name} ({_megabytes(asset.size)}) -- "
                  "nothing to carry the CUDA runtime over from")
        case ((newest, *_), Absent(name)):
            print(f"Would carry the CUDA runtime over from {newest.name}; this release "
                  f"carries no {name} yet")
        case ((newest, *_), Present(asset)):
            print(f"Would carry the CUDA runtime over from {newest.name} instead of "
                  f"downloading {_megabytes(asset.size)}")

    print(f"Would install into {target}")


def _assemble(root: Path, installed: Sequence[Release], latest: Latest,
              target: Path) -> None:
    """The release, assembled beside the others and moved in once it verifies."""
    staging = root / f".staging-{target.name}"
    temporary = Path(tempfile.gettempdir()) / f"llamacpp-update-{latest.build}"

    files.fresh(staging)
    files.fresh(temporary)

    try:
        print()
        print(f"Downloading {latest.binaries.name} "
              f"({_megabytes(latest.binaries.size)}) ...")
        archive = temporary / latest.binaries.name
        _download(latest.binaries, archive)

        print("Extracting ...")
        files.unpack(archive, staging)
        _flatten(staging)

        if not files.exists(staging / SERVER):
            raise Refusal(f"The archive carries no {SERVER} where one was expected, so "
                          "nothing was installed.")

        _runtime(staging, root, installed, latest, temporary)

        print("Verifying ...")
        said = _version(staging / SERVER)
        if not releases.verifies(said, latest.build):
            raise Refusal("The unpacked release is not the build it was downloaded as, "
                          f"so nothing was installed. --version said: {said.strip()!r}")
        print(said.strip())

        files.remove(target)
        files.move(staging, target)
        print(f"Installed {target.name}")
    finally:
        files.discard(temporary)
        files.discard(staging)


def _runtime(staging: Path, root: Path, installed: Sequence[Release], latest: Latest,
             temporary: Path) -> None:
    """The CUDA runtime: carried over from the release already here, or downloaded.

    Carrying it over is the ordinary case and saves nearly four hundred megabytes, the
    runtime being the same file for every release of a CUDA version. It is also what
    makes a release installable whose own runtime archive is not up yet: the binaries
    of a release are published before it.
    """
    match installed:
        case (newest, *_):
            carried = releases.missing(files.named(staging, ".dll"),
                                       files.named(root / newest.name, ".dll"))
            for name in carried:
                files.copy(root / newest.name / name, staging)

            if carried:
                print(f"Carried {len(carried)} CUDA runtime file(s) over from "
                      f"{newest.name}")
                return

    match latest.runtime:
        case Absent(name):
            raise Refusal("There is no CUDA runtime to carry over, and "
                          f"b{latest.build} carries no {name}.")
        case Present(asset):
            print(f"Downloading {asset.name} ({_megabytes(asset.size)}) ...")
            archive = temporary / asset.name
            _download(asset, archive)
            files.unpack(archive, staging)


def _flatten(staging: Path) -> None:
    """Some archives wrap everything in a single folder. Lift it, so that a release is
    always the server and its libraries directly under the one directory."""
    if files.exists(staging / SERVER):
        return

    match files.directories(staging):
        case (only,) if files.exists(staging / only / SERVER):
            for entry in files.contents(staging / only):
                files.move(entry, staging / entry.name)
            files.remove(staging / only)


def _prune(root: Path, keep: int) -> None:
    """The releases past keeping, removed -- except one a server is running from."""
    running = session.executing(SERVER)
    installed = releases.releases(files.directories(root))
    serving = {one.name for one in installed
               if str(root / one.name).lower() in running}

    pruned = releases.prune(installed, keep, serving)

    for release in pruned.spared:
        print(f"Keeping {release.name}: a server is running from it.")

    for release in pruned.remove:
        path = root / release.name
        freed = files.size(path)
        files.remove(path)
        print(f"Removed {release.name} ({_megabytes(freed)})")


def _server(root: Path) -> Path:
    """What the router will run: the server in the newest release unpacked here."""
    for release in releases.releases(files.directories(root)):
        server = root / release.name / SERVER
        if files.exists(server):
            return server

    raise Refusal(f"no llama.cpp release with {SERVER} under {root}")


def _version(server: Path) -> str:
    """What the unpacked server says it is. It says it on the error stream."""
    said = proc.run((str(server), "--version"))
    return said.out + said.err


def _answer() -> object:
    """What the API says has been published.

    It is the only source for what a release carries, so a run that cannot reach it
    installs nothing rather than guessing at the names of the archives.
    """
    request = urllib.request.Request(upstream.RELEASES_URL, headers=HEADERS)

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as answer:
            return json.loads(answer.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as unreachable:
        raise Refusal(
            f"{upstream.RELEASES_URL} did not answer: {unreachable}\n"
            "That is where the published releases are listed, so nothing was "
            "installed.") from None


def _download(asset: Asset, into: Path) -> None:
    """One asset, whole.

    Whole is checked rather than assumed: a download cut short by the network arrives
    as a file like any other, and a truncated archive extracts into a release missing
    whatever was at the end of it.
    """
    request = urllib.request.Request(asset.url, headers=HEADERS)

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as answer:
            with into.open("wb") as file:
                shutil.copyfileobj(answer, file)
    except (urllib.error.URLError, OSError) as unreachable:
        raise Refusal(f"{asset.name} did not download: {unreachable}") from None

    arrived = Bytes(into.stat().st_size)
    if arrived != asset.size:
        raise Refusal(f"{asset.name} arrived as {arrived} bytes where the release says "
                      f"{asset.size}. Treating that as a failed download; nothing was "
                      "installed.")


def _megabytes(size: Bytes) -> str:
    return f"{size / _MEGABYTE:.1f} MB"
