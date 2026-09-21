"""The llama.cpp release the router runs: installed here, or brought up to date.

The loop is here and it decides nothing. upstream.py reads what the project published
and says which release can be installed, releases.py says what is here already and what
is past keeping, and this asks, downloads, unpacks and removes.

A release is assembled in a staging directory beside the others and moved into place
only once the server in it says it is the build that was asked for, and only then is it
recorded in the settings file as the build to run. Nothing part-way downloaded is ever
the release something runs.

This is the one place a release is put on a machine, the router's or a slave's alike:
the two have to run the same build to talk at all, and one command on both, told the
same build, is how they come to.
"""

import json
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from . import config, devices, files, proc, releases, session, upstream, workspace
from .config import ConfigError
from .refusal import Refusal
from .releases import Recorded, Release, Running, Unrecorded
from .rpc import WORKER
from .serving import SERVER
from .units import Bytes
from .upstream import (Absent, AsPinned, Asked, Asset, BeyondDriver, Choice, Cuda,
                       Exactly, Latest, Newest, NewestBuild, Present, Published,
                       Unpublished)

# GitHub asks every caller to say what it is, and turns away one that does not.
HEADERS = {"User-Agent": "llamacpp-local-updater"}

# How long to wait for an answer. Not a deadline for a download: it bounds each read,
# and a download that is arriving is answering.
TIMEOUT = 120

# What the API answers for a tag nobody published.
NOT_FOUND = 404

# Left in a release installed by naming its build. Pruning passes it over: it was put
# there by hand, and it goes the same way.
BY_HAND = "installed-by-hand.txt"
BY_HAND_SAYS = ("Installed with python -m cm.install llamacpp --build. Pruning leaves this "
                "release alone; remove it by hand.\n")

_MEGABYTE = 1048576


def install(settings: Path, asked: Asked, check: bool, force: bool) -> None:
    """A release put here and recorded in the settings file as the build to run: the
    build asked for, or the newest published for the CUDA version this machine takes.

    A release already here is not downloaded again. Asking for it is how a machine is
    moved back onto a build it has, and recording it is all that takes.
    """
    _settings(settings)
    read = config.installing(files.read(settings))
    root = workspace.engines()

    installed = releases.releases(files.directories(root))
    print(f"llama.cpp under {root}")
    print("Installed: " + (", ".join(one.name for one in installed) or "none"))
    print(f"Runs:      {_runs(read.build)}")

    driver = devices.driver()
    print(f"Driver:    {driver.version}, CUDA {driver.cuda.version}")
    for one in devices.attached():
        print(f"Card {one.index}:    {one.card.name}, compute capability "
              f"{one.capability.major}.{one.capability.minor}")

    published = _published(asked)
    choice = upstream.choose(read.cuda, published, driver.cuda)
    cuda = choice.cuda
    print(_chosen(choice, driver.cuda))

    latest = upstream.latest(published, cuda)
    if latest.incomplete:
        print("Skipped (archive not uploaded yet): "
              + ", ".join(f"b{build}" for build in latest.incomplete))
    print(f"Build:     b{latest.build} (published {latest.when})")

    target = root / upstream.directory(latest.build, cuda)
    here = any(one.name == target.name for one in installed) and not force
    alike = releases.built_against(installed, cuda)

    if here and read.build == Recorded(latest.build) and not isinstance(asked, Exactly):
        print(f"{target.name} is here and recorded as the build to run. Nothing to do.")
        return

    if check:
        _would(installed, alike, latest, target, here)
        return

    print()
    if here:
        print(f"{target.name} is already here: nothing to download.")
    else:
        files.ensure(root)
        _assemble(root, alike, latest, target)

    if isinstance(asked, Exactly):
        files.write(target / BY_HAND, BY_HAND_SAYS)
    _record(settings, latest.build)
    _prune(root, read.keep_releases, target.name)

    print()
    print(f"Runs from now on: {target}")
    print("Restart what runs from it: python -m cm.router start on the machine with the "
          "router, python -m cm.slave start on a machine lending its card.")


def _published(asked: Asked) -> tuple[Published, ...]:
    """The releases to choose from: the recent ones, or the one build asked for,
    however old."""
    match asked:
        case NewestBuild():
            return upstream.published(
                _answer(upstream.RELEASES_URL,
                        absent=f"{upstream.RELEASES_URL} lists nothing."))
        case Exactly(number):
            return upstream.published(
                [_answer(upstream.tagged(number),
                         absent=f"No llama.cpp release is tagged b{number}. The "
                                "published ones are at "
                                f"https://github.com/{upstream.REPO}/releases")])


def _runs(build: Running) -> str:
    """Which build this machine runs now, as its settings file has it."""
    match build:
        case Recorded(number):
            return f"b{number}, as the settings file records it"
        case Unrecorded():
            return "the newest release here: the settings file records no build yet"


def _chosen(choice: Choice, driver: Cuda) -> str:
    """Which CUDA version the release is for, and why that one."""
    match choice:
        case Newest(cuda):
            return f"CUDA {cuda.version}: the newest published that this driver runs"
        case AsPinned(cuda):
            return f"CUDA {cuda.version}: as cuda_version pins it"
        case Unpublished(pinned, cuda):
            return (f"CUDA {cuda.version}: cuda_version pins {pinned.version}, which no "
                    "release is built for lately, so the newest this driver runs instead")
        case BeyondDriver(pinned, cuda):
            return (f"CUDA {cuda.version}: cuda_version pins {pinned.version}, and this "
                    f"driver runs {driver.version} at most, so the newest it runs instead")


def _settings(settings: Path) -> None:
    """The settings file, made from the one that ships where this machine has none yet.

    Copied rather than refused: somebody who has just been handed the scripts should not
    have to write a file by hand to use them. Nothing in the copy has to be answered
    before a release is installed. Where the models are is asked by scan, the first
    command that needs them, and a machine lending its card never needs them at all.
    """
    if files.exists(settings):
        return

    template = workspace.template()
    if not files.exists(template):
        raise ConfigError(f"{settings.name} not found at {settings}, and neither is "
                          f"the file it would be copied from: {template}")

    files.write(settings, files.read(template))

    print(f"Copied {template.name} to {settings}")
    print("On the machine with the router it names no model yet: python -m cm.scan "
          "writes an entry for each one in the library.")
    print()


def _record(settings: Path, number: int) -> None:
    """The build written into the settings file as the one everything here runs."""
    was = files.read(settings)
    now = config.recording_the_build(was, number)
    if now != was:
        files.replace(settings, now)

    print(f"Recorded in {settings.name}: {config.BUILD} = {number}")


def _would(installed: Sequence[Release], alike: Sequence[Release], latest: Latest,
           target: Path, here: bool) -> None:
    """What installing would take, with nothing downloaded and nothing written. alike
    is what is installed of the same CUDA version, which is all a runtime is carried
    over from."""
    print()
    if here:
        print(f"{target.name} is already here: would download nothing, and record "
              f"b{latest.build} as the build to run")
        return

    match installed:
        case (newest, *_):
            print(f"Update available: {newest.name} -> b{latest.build}")
        case _:
            print(f"Nothing installed yet: would install b{latest.build}.")

    print(f"Would download {latest.binaries.name} ({_megabytes(latest.binaries.size)})")

    match (tuple(alike), latest.runtime):
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

    print(f"Would install into {target}, and record b{latest.build} as the build to run")


def _assemble(root: Path, alike: Sequence[Release], latest: Latest,
              target: Path) -> None:
    """The release, assembled beside the others and moved in once it verifies. alike is
    what is installed of the same CUDA version."""
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

        _runtime(staging, root, alike, latest, temporary)

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


def _runtime(staging: Path, root: Path, alike: Sequence[Release], latest: Latest,
             temporary: Path) -> None:
    """The CUDA runtime: carried over from a release of the same CUDA version already
    here, or downloaded.

    Carrying it over is the ordinary case and saves nearly four hundred megabytes, the
    runtime being the same file for every release of a CUDA version. It is also what
    makes a release installable whose own runtime archive is not up yet: the binaries
    of a release are published before it.
    """
    match tuple(alike):
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


def _prune(root: Path, keep: int, runs: str) -> None:
    """The releases past keeping, removed -- except the one that runs, one a server is
    running from, and one installed by naming its build, which goes by hand."""
    running = session.executing(SERVER) | session.executing(WORKER)
    installed = releases.releases(files.directories(root))
    serving = {one.name for one in installed
               if str(root / one.name).lower() in running}
    held = {runs} | {one.name for one in installed
                     if files.exists(root / one.name / BY_HAND)}

    pruned = releases.prune(installed, keep, serving, held)

    for release in pruned.spared:
        print(f"Keeping {release.name}: a server is running from it.")

    for release in pruned.remove:
        path = root / release.name
        freed = files.size(path)
        files.remove(path)
        print(f"Removed {release.name} ({_megabytes(freed)})")


def _version(server: Path) -> str:
    """What the unpacked server says it is. It says it on the error stream."""
    said = proc.run((str(server), "--version"))
    return said.out + said.err


def _answer(url: str, absent: str) -> object:
    """What the API says has been published there.

    It is the only source for what a release carries, so a run that cannot reach it
    installs nothing rather than guessing at the names of the archives. absent is what
    to say where it answers that there is nothing there at all.
    """
    request = urllib.request.Request(url, headers=HEADERS)

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as answer:
            return json.loads(answer.read().decode("utf-8"))
    except urllib.error.HTTPError as refused:
        if refused.code == NOT_FOUND:
            raise Refusal(absent) from None
        raise Refusal(f"{url} did not answer: {refused}\n"
                      "That is where the published releases are read, so nothing was "
                      "installed.") from None
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as unreachable:
        raise Refusal(
            f"{url} did not answer: {unreachable}\n"
            "That is where the published releases are read, so nothing was "
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
