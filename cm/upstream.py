"""What the llama.cpp project has published, and which release to install.

A release carries some thirty files -- the Windows binaries for every CUDA version
built, the CUDA runtime beside them, the archives for the other systems, the sources.
Two of them are read here, both named after the CUDA version this machine runs.

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

from .units import Bytes

REPO = "ggml-org/llama.cpp"

# One page, thirty releases: more than a day of builds, and a day of them going by with
# none carrying this CUDA version is not something to page through -- it is something to
# report.
RELEASES_URL = f"https://api.github.com/repos/{REPO}/releases?per_page=30"

# A build's own tag. A release tagged anything else belongs to something other than the
# per-build stream this installs from.
_TAG = re.compile(r"^b(\d+)$")


class Unavailable(Exception):
    """Nothing published can be installed, with the one line saying why."""


@dataclass(frozen=True)
class Cuda:
    """The CUDA version the binaries were built against, as the archives spell it."""

    version: str


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
