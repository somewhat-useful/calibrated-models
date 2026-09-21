"""What the llama.cpp project has published, and which release to install.

A release carries some thirty files -- the Windows binaries for every CUDA version
built, the CUDA runtime beside them, the archives for the other systems, the sources.
Two of them are installed, both named after the CUDA version chosen for this machine.

Which CUDA version that is, is not written down anywhere as a constant. The project
moves its builds from one version to the next and stops publishing the old one, so a
version fixed here would stop installing anything the day it moved. What is published
is read instead, and the newest of it that runs on this machine is taken: on this
driver, and on every card here.

Which release to install is not simply the newest published. The assets of a release
appear over several minutes, so the newest tag frequently exists with the Windows
archive still missing; a release carrying no archive for this CUDA version is not an
update yet, and the one below it is. Nor is it what /releases/latest answers: every
b-numbered build is published as a prerelease, and that endpoint answers with the
newest release that is not one, which is an unrelated version tag.

Parsed JSON in, values out. Nothing is fetched and nothing is written here.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .architectures import Architectures, finished
from .cuda import Cuda
from .machine import Attached
from .units import Bytes

REPO = "ggml-org/llama.cpp"

# One page, thirty releases: more than a day of builds, and a day of them going by with
# none carrying this CUDA version is not something to page through -- it is something to
# report.
RELEASES_URL = f"https://api.github.com/repos/{REPO}/releases?per_page=30"


def tagged(number: int) -> str:
    """Where the API answers for one build's release, however far back it was
    published."""
    return f"https://api.github.com/repos/{REPO}/releases/tags/b{number}"


def source(number: int, path: str) -> str:
    """Where one file of the project's source is read, as it stood at a build's tag."""
    return f"https://raw.githubusercontent.com/{REPO}/b{number}/{path}"

# A build's own tag. A release tagged anything else belongs to something other than the
# per-build stream this installs from.
_TAG = re.compile(r"^b(\d+)$")

# The Windows archive of a build, and the CUDA version it was built against.
_BINARIES = re.compile(r"^llama-b(\d+)-bin-win-cuda-(\d+)\.(\d+)-x64\.zip$")


class Unavailable(Exception):
    """Nothing published can be installed, with the one line saying why."""


@dataclass(frozen=True)
class Driver:
    """The NVIDIA driver on this machine: its own version, 616.56, and the newest CUDA
    version it runs."""

    version: str
    cuda: Cuda


@dataclass(frozen=True)
class NewestRunnable:
    """No CUDA version pinned: the newest one published that this machine's driver
    runs."""


# What a settings file asks for with cuda_version, or without it.
Wanted = Cuda | NewestRunnable


@dataclass(frozen=True)
class Runs:
    """A build this driver runs, on every card here."""


@dataclass(frozen=True)
class DriverTooOld:
    """The driver belongs to an older major CUDA version than the build, and nothing on
    a Windows machine makes that run."""


@dataclass(frozen=True)
class NotReady:
    """The driver is older than the build, and these cards have only PTX in it: code
    the driver finishes compiling when a model loads, which a driver older than the
    build cannot."""

    cards: tuple[Attached, ...]


# Why a build would not run here.
Hindrance = DriverTooOld | NotReady

# Whether a build runs here.
Verdict = Runs | Hindrance


@dataclass(frozen=True)
class Newest:
    """Nothing pinned, and this the newest version published: it runs here."""

    cuda: Cuda


@dataclass(frozen=True)
class HeldBack:
    """Nothing pinned, and the newest version published would not run here. This, the
    newest that does, is installed instead, and a newer driver is what the newest
    needs."""

    newest: Cuda
    cuda: Cuda
    why: Hindrance


@dataclass(frozen=True)
class AsPinned:
    """The version the settings file pins, published, and it runs here."""

    cuda: Cuda


@dataclass(frozen=True)
class Unpublished:
    """The pinned version is not among the builds published lately, and this, the
    newest that runs here, stands in for it."""

    pinned: Cuda
    cuda: Cuda


@dataclass(frozen=True)
class PinnedHeldBack:
    """The pinned version would not run here, and this, the newest that does, stands in
    for it."""

    pinned: Cuda
    cuda: Cuda
    why: Hindrance


# Which CUDA version the release installed is built against, and why that one.
Choice = Newest | HeldBack | AsPinned | Unpublished | PinnedHeldBack


@dataclass(frozen=True)
class NewestBuild:
    """The newest build published for the CUDA version this machine takes."""


@dataclass(frozen=True)
class Exactly:
    """One build, by its number: the one another machine runs, say, so that a router and
    a slave run the same."""

    number: int


# Which build install puts in place.
Asked = NewestBuild | Exactly


@dataclass(frozen=True)
class Asset:
    """One file published with a release.

    The size is what the release says the file is, and it is what a finished download
    is measured against. A release that states no size states zero, which no download
    matches -- the direction to be wrong in.
    """

    name: str
    size: Bytes
    url: str


@dataclass(frozen=True)
class Present:
    """An asset the release carries."""

    asset: Asset


@dataclass(frozen=True)
class Absent:
    """An asset it does not carry, and what it would have been called."""

    name: str


Carried = Present | Absent


@dataclass(frozen=True)
class Published:
    """A release as the project published it."""

    build: int
    when: str
    assets: tuple[Asset, ...]


@dataclass(frozen=True)
class Latest:
    """The newest release that can be installed, and what installing it takes.

    `incomplete` are the newer builds passed over because their archive is not up yet.
    They are worth saying out loud: an update that stops one build short is the ordinary
    case minutes after a build is tagged, not a fault.
    """

    build: int
    when: str
    binaries: Asset
    runtime: Carried
    incomplete: tuple[int, ...]


def binaries(build: int, cuda: Cuda) -> str:
    """What the Windows archive of a build is called."""
    return f"llama-b{build}-bin-win-cuda-{cuda.version}-x64.zip"


def runtime(cuda: Cuda) -> str:
    """What the CUDA runtime archive is called.

    The same file for every release of a CUDA version, and nearly four hundred
    megabytes of it, which is why it is carried over from the release already here
    rather than downloaded again.
    """
    return f"cudart-llama-bin-win-cuda-{cuda.version}-x64.zip"


def directory(build: int, cuda: Cuda) -> str:
    """What a release is called once it is unpacked here.

    The build number leads, because the highest one present is the one everything runs:
    unpacking is what makes a release current, and nothing has to be repointed at it.
    """
    return f"b{build}-cuda{cuda.version}"


def published(answer: object) -> tuple[Published, ...]:
    """The per-build releases in what the API answered, newest first.

    Whatever else the answer holds is not one of them: another tagging scheme, a field
    missing, an entry that is not a release at all. None of that is a failure to report
    -- it is a release this does not install from.
    """
    if not isinstance(answer, list):
        return ()

    found = []
    for entry in answer:
        if not isinstance(entry, dict):
            continue

        tag = _TAG.match(str(entry.get("tag_name", "")))
        if tag is None:
            continue

        found.append(Published(build=int(tag[1]),
                               when=str(entry.get("published_at", "")),
                               assets=_assets(entry.get("assets"))))

    return tuple(sorted(found, key=lambda one: one.build, reverse=True))


def supported(encoded: int) -> Cuda:
    """What cuDriverGetVersion answers, as a version: 1000 * major + 10 * minor, so 13040
    is 13.4."""
    return Cuda(encoded // 1000, encoded % 1000 // 10)


def choose(wanted: Wanted, releases: Sequence[Published], driver: Cuda,
           cards: Sequence[Attached], known: Architectures) -> Choice:
    """Which CUDA version to install a release for: the newest published that runs on
    this driver and every card here.

    Where the newest published does not run, the newest that does is taken and the
    choice says why, so the driver can be brought up to it -- a machine refusing to
    install because its driver lags is worse off than one running the version before.

    A pinned version is taken where it is published and runs here, and stood in for the
    same way where it does not: the pin is for keeping machines alike, and a machine
    that installs nothing because the project stopped building that version is worse
    off than one running the next.
    """
    published = flavours(releases)
    running = tuple(one for one in published
                    if isinstance(verdict(one, driver, cards, known), Runs))

    match running:
        case ():
            raise Unavailable(_nothing_runs(published, driver, cards, known, releases))
        case (newest, *_):
            pass

    match wanted:
        case NewestRunnable():
            match verdict(published[0], driver, cards, known):
                case Runs():
                    return Newest(newest)
                case DriverTooOld() | NotReady() as why:
                    return HeldBack(published[0], newest, why)
        case Cuda() as pinned if pinned not in published:
            return Unpublished(pinned, newest)
        case Cuda() as pinned:
            match verdict(pinned, driver, cards, known):
                case Runs():
                    return AsPinned(pinned)
                case DriverTooOld() | NotReady() as why:
                    return PinnedHeldBack(pinned, newest, why)


def verdict(build: Cuda, driver: Cuda, cards: Sequence[Attached],
            known: Architectures) -> Verdict:
    """Whether a build for this CUDA version runs on this driver, on every card here.

    NVIDIA's CUDA Compatibility guide: a driver runs a build for its own CUDA version or
    an older one. One of the same major version runs a newer build too, but only the
    code compiled for a card's own architecture; PTX from a newer toolkit is beyond it.
    A driver of an older major version runs nothing newer at all, short of a
    compatibility package that is not made for Windows.
    """
    if build <= driver:
        return Runs()
    if build.major > driver.major:
        return DriverTooOld()

    lacking = tuple(one for one in cards
                    if one.capability not in finished(build, known))
    return NotReady(lacking) if lacking else Runs()


def hindered(why: Hindrance, build: Cuda, driver: Cuda) -> str:
    """Why a build for this version would not run here, as the end of a sentence."""
    match why:
        case DriverTooOld():
            return (f"needs a CUDA {build.major} driver, and this one is CUDA "
                    f"{driver.version}")
        case NotReady(cards):
            names = ", ".join(f"{one.card.name} ({one.capability.major}."
                              f"{one.capability.minor})" for one in cards)
            return (f"carries no finished code for {names}, which needs a CUDA "
                    f"{build.version} driver to run it, and this one is CUDA "
                    f"{driver.version}")


def flavours(releases: Sequence[Published]) -> tuple[Cuda, ...]:
    """Every CUDA version a Windows build was published for, the newest first."""
    found = set()
    for release in releases:
        for asset in release.assets:
            named = _BINARIES.match(asset.name)
            if named is not None and int(named[1]) == release.build:
                found.add(Cuda(int(named[2]), int(named[3])))

    return tuple(sorted(found, reverse=True))


def _nothing_runs(published: Sequence[Cuda], driver: Cuda, cards: Sequence[Attached],
                  known: Architectures, releases: Sequence[Published]) -> str:
    """Why no version can be chosen: none is published, or none runs here."""
    match published:
        case (newest, *_):
            match verdict(newest, driver, cards, known):
                case DriverTooOld() | NotReady() as why:
                    return (f"Nothing published lately runs here. The newest, CUDA "
                            f"{newest.version}, {hindered(why, newest, driver)}.\n"
                            "Update the NVIDIA driver, and run this again.")
                case Runs():
                    pass

    seen = ", ".join(f"b{one.build}" for one in releases[:5]) or "none"
    return ("No release carries a Windows CUDA archive, "
            "llama-b<number>-bin-win-cuda-<version>-x64.zip.\n"
            "\n"
            f"Newest tags seen: {seen}\n"
            "\n"
            "The release naming has changed and this needs revisiting. The published "
            f"archives are at https://github.com/{REPO}/releases")


def latest(releases: Sequence[Published], cuda: Cuda) -> Latest:
    """The newest release that actually carries the archive for this CUDA version."""
    incomplete = []

    for release in sorted(releases, key=lambda one: one.build, reverse=True):
        match _carried(release, binaries(release.build, cuda)):
            case Present(asset):
                return Latest(build=release.build,
                              when=release.when,
                              binaries=asset,
                              runtime=_carried(release, runtime(cuda)),
                              incomplete=tuple(incomplete))
            case Absent():
                incomplete.append(release.build)

    seen = ", ".join(f"b{one.build}" for one in releases[:5]) or "none"
    raise Unavailable(
        f"No release carries llama-b<number>-bin-win-cuda-{cuda.version}-x64.zip.\n"
        f"\n"
        f"Newest tags seen: {seen}\n"
        f"\n"
        f"Either cuda_version in the settings file names a flavour that is no longer "
        f"built, or the release naming has changed and this needs revisiting. The "
        f"published archives are at https://github.com/{REPO}/releases")


def _carried(release: Published, name: str) -> Carried:
    for asset in release.assets:
        if asset.name == name:
            return Present(asset)

    return Absent(name)


def _assets(listed: object) -> tuple[Asset, ...]:
    if not isinstance(listed, list):
        return ()

    return tuple(_asset(one) for one in listed if isinstance(one, dict))


def _asset(one: Mapping[str, object]) -> Asset:
    size = one.get("size")

    return Asset(name=str(one.get("name", "")),
                 size=Bytes(size if isinstance(size, int)
                            and not isinstance(size, bool) else 0),
                 url=str(one.get("browser_download_url", "")))
