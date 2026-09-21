"""Which cards a published llama.cpp build carries finished code for.

A CUDA build holds each card's code in one of two forms: finished machine code for that
card's architecture, or PTX, which the driver compiles when a model loads. A driver of
the build's own major version runs a newer build than itself only where the code is
finished -- PTX from a newer toolkit is beyond it -- so which form a card gets decides
whether a build runs on a driver that lags it.

Nothing a release publishes says which. What does is the project's own source at that
build's tag: the Windows builds name no architectures of their own, and so take ggml's
defaults, which ggml/src/ggml-cuda/CMakeLists.txt appends one CUDA version at a time.
That file is read here rather than copied, so a card the project adds is known the day
it is added. What cannot be read is not guessed at: the caller holds every build to
what the driver runs on any card instead.

Text in, values out. Nothing is fetched here.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .cuda import Cuda
from .machine import Capability


@dataclass(frozen=True)
class From:
    """From this CUDA version on."""

    cuda: Cuda


@dataclass(frozen=True)
class Before:
    """Below this CUDA version."""

    cuda: Cuda


# A CUDA version condition an architecture is compiled under.
Bound = From | Before


@dataclass(frozen=True)
class Compiled:
    """One architecture compiled to finished code, and the CUDA versions it is compiled
    under: every one of the bounds holds."""

    capability: Capability
    bounds: tuple[Bound, ...]


@dataclass(frozen=True)
class Architectures:
    """What the release builds compile to finished code."""

    compiled: tuple[Compiled, ...]


@dataclass(frozen=True)
class Unread:
    """The source could not be read for it, and the one line saying why."""

    why: str


# What says nothing is finished: every build is then held to what the driver runs on
# any card, which is the one thing true whatever a build carries.
NOTHING_KNOWN = Architectures(compiled=())

# Where the source says it, relative to the root of the project at a build's tag.
SOURCE = "ggml/src/ggml-cuda/CMakeLists.txt"
WORKFLOW = ".github/workflows/release.yml"

# The block ggml's defaults sit in: it applies only where a build names none itself.
_DEFAULTS = "if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)"

_IF = re.compile(r"^if\s*\((.*)\)$")
_ELSE = re.compile(r"^else\s*\(\s*\)$")
_ENDIF = re.compile(r"^endif\s*\(\s*\)$")
_APPEND = re.compile(r"^list\s*\(\s*APPEND\s+CMAKE_CUDA_ARCHITECTURES\s+([^)]*)\)$")
_VERSION = re.compile(
    r'^CUDAToolkit_VERSION\s+(VERSION_GREATER_EQUAL|VERSION_LESS)\s+"(\d+)(?:\.(\d+))?"$')

# 86-real, 120a-real, 90-virtual, 80: the architecture's digits, the minor one last, an
# architecture-specific or family suffix, and what is compiled. No suffix is both.
_ARCHITECTURE = re.compile(r"^(\d+)(\d)[af]?(-real|-virtual)?$")

# The job that builds the Windows CUDA archives, as the workflow names it, and where
# the next job starts.
_WINDOWS_JOB = re.compile(r"^  windows-cuda:\s*$", re.MULTILINE)
_NEXT_JOB = re.compile(r"^  [A-Za-z0-9_-]+:\s*$", re.MULTILINE)


@dataclass(frozen=True)
class _Applies:
    """The Windows builds name no architectures of their own, so the defaults are
    theirs."""


@dataclass(frozen=True)
class _Native:
    """Inside the branch a build compiling for the machine it is built on takes."""


@dataclass(frozen=True)
class _NotNative:
    """Inside the branch every published build takes: they are built with GGML_NATIVE
    off, for machines other than the one building them."""


@dataclass(frozen=True)
class _Unknown:
    """Inside a condition this does not read."""

    said: str


_Frame = Bound | _Native | _NotNative | _Unknown


def read(source: str, workflow: str) -> Architectures | Unread:
    """What the release builds compile to finished code, as the source at their tag
    says, or why that could not be read.

    The workflow is read first: the defaults are only what a build takes where it names
    no architectures itself, and a Windows build that started naming them would make
    the defaults say nothing about it.
    """
    match _names_none(workflow):
        case Unread() as unread:
            return unread
        case _Applies():
            pass

    return _defaults(source)


def finished(build: Cuda, known: Architectures) -> frozenset[Capability]:
    """The cards a build for this CUDA version carries finished code for."""
    return frozenset(one.capability for one in known.compiled
                     if all(_holds(bound, build) for bound in one.bounds))


def _holds(bound: Bound, build: Cuda) -> bool:
    match bound:
        case From(cuda):
            return build >= cuda
        case Before(cuda):
            return build < cuda


def _names_none(workflow: str) -> Unread | _Applies:
    """Whether the Windows CUDA builds name architectures of their own."""
    start = _WINDOWS_JOB.search(workflow)
    if start is None:
        return Unread(f"{WORKFLOW} has no windows-cuda job")

    following = _NEXT_JOB.search(workflow, start.end())
    job = workflow[start.end():following.start() if following else len(workflow)]
    if "CMAKE_CUDA_ARCHITECTURES" in job:
        return Unread(f"the windows-cuda job in {WORKFLOW} names architectures of its own")

    return _Applies()


def _defaults(source: str) -> Architectures | Unread:
    """The architectures the defaults block compiles to finished code."""
    lines = [line.strip() for line in source.splitlines()]
    try:
        opening = lines.index(_DEFAULTS)
    except ValueError:
        return Unread(f"{SOURCE} has no '{_DEFAULTS}' block")

    stack: list[_Frame] = []
    compiled: list[Compiled] = []

    for line in lines[opening + 1:]:
        if not line or line.startswith("#"):
            continue

        if (condition := _IF.match(line)) is not None:
            stack.append(_frame(condition[1].strip()))
        elif _ELSE.match(line):
            if not stack:
                return Unread(f"{SOURCE} has an else() this cannot place")
            stack[-1] = _otherwise(stack[-1])
        elif _ENDIF.match(line):
            if not stack:
                break
            stack.pop()
        elif line.startswith("elseif"):
            return Unread(f"{SOURCE} chooses architectures with elseif(), which this "
                          "does not read")
        elif (appended := _APPEND.match(line)) is not None:
            match _under(stack):
                case Unread() as unread:
                    return unread
                case (*bounds,):
                    match _entries(appended[1].split(), tuple(bounds)):
                        case Unread() as unread:
                            return unread
                        case (*read_here,):
                            compiled += read_here
        elif "CMAKE_CUDA_ARCHITECTURES" in line and not (
                stack and isinstance(stack[-1], _Native)):
            return Unread(f"{SOURCE} sets the architectures in a way this does not "
                          f"read: {line}")
    else:
        return Unread(f"{SOURCE} does not close its '{_DEFAULTS}' block")

    if not compiled:
        return Unread(f"{SOURCE} compiles no architecture to finished code, as far as "
                      "this reads it")

    return Architectures(compiled=tuple(compiled))


def _frame(condition: str) -> _Frame:
    """What an if() opens."""
    version = _VERSION.match(condition)
    if version is not None:
        cuda = Cuda(int(version[2]), int(version[3] or 0))
        return From(cuda) if version[1] == "VERSION_GREATER_EQUAL" else Before(cuda)

    if condition.startswith("GGML_NATIVE"):
        return _Native()

    return _Unknown(condition)


def _otherwise(frame: _Frame) -> _Frame:
    """What the else() of that if() is inside."""
    match frame:
        case From(cuda):
            return Before(cuda)
        case Before(cuda):
            return From(cuda)
        case _Native():
            return _NotNative()
        case _NotNative():
            return _Native()
        case _Unknown(said):
            return _Unknown(f"not ({said})")


def _under(stack: Sequence[_Frame]) -> tuple[Bound, ...] | Unread:
    """The CUDA versions an architecture appended here is compiled under."""
    bounds: list[Bound] = []
    for frame in stack:
        match frame:
            case From() | Before():
                bounds.append(frame)
            case _NotNative():
                pass
            case _Native():
                return Unread(f"{SOURCE} appends an architecture for native builds only")
            case _Unknown(said):
                return Unread(f"{SOURCE} appends an architecture under a condition this "
                              f"does not read: {said}")

    return tuple(bounds)


def _entries(words: Sequence[str], bounds: tuple[Bound, ...]
             ) -> tuple[Compiled, ...] | Unread:
    """The architectures in one list(APPEND ...), the finished ones kept."""
    kept = []
    for word in words:
        named = _ARCHITECTURE.match(word)
        if named is None:
            return Unread(f"{SOURCE} names an architecture this does not read: {word}")
        if named[3] == "-virtual":
            continue
        kept.append(Compiled(capability=Capability(int(named[1]), int(named[2])),
                             bounds=bounds))

    return tuple(kept)
