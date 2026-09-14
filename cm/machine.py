"""What this machine is, in the numbers a placement and a preset need.

Nothing here reads a device. The figures arrive from devices.py already measured, and
what is computed from them is computed the same way whoever asks.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NewType

from .nonempty import NonEmpty
from .units import Mib


class UnreadableDevice(Exception):
    """A device answered something this cannot read, with the one line saying what."""


@dataclass(frozen=True)
class Card:
    """A video card a model is placed on."""

    name: str
    total: Mib


# The number a card goes by: nvidia-smi's index, and the n in llama.cpp's CUDAn. The two
# agree because every program this starts is told to count cards the way nvidia-smi does.
CudaIndex = NewType("CudaIndex", int)


@dataclass(frozen=True, order=True)
class Capability:
    """How recent a generation of card this is, as CUDA numbers it: 12.0 for an RTX 5070
    Ti, 7.5 for an RTX 2070. A later generation is the faster card."""

    major: int
    minor: int


@dataclass(frozen=True)
class Installed:
    """A card in this machine, and what placing a model across several of them needs.

    drives_display is whether a monitor is plugged into it. A card drawing a desktop is
    one somebody may be working at, and a placement leaves it room for that.
    """

    index: CudaIndex
    card: Card
    capability: Capability
    drives_display: bool


@dataclass(frozen=True)
class Occupancy:
    """A card and how much of it is free right now."""

    card: Card
    free: Mib


@dataclass(frozen=True)
class Core:
    """One physical core: how fast a class it belongs to, and how many threads it runs.

    A hybrid processor sorts its cores into efficiency classes, and a higher class is
    the faster one. On an i7-13700F the eight performance cores are class 1 with two
    logical processors each, and the eight efficient cores are class 0 with one.
    """

    efficiency_class: int
    logical: int


@dataclass(frozen=True)
class Machine:
    """The cards, the system memory, and the cores. A machine has at least one core."""

    cards: NonEmpty[Installed]
    ram: Mib
    cores: tuple[Core, ...]

    def __post_init__(self) -> None:
        if not self.cores:
            raise ValueError("a machine with no cores is not a machine")


@dataclass(frozen=True)
class Fixed:
    """A size the settings file wrote down: 32, 32G and 32Gb all mean 32 gibibytes."""

    size: Mib


@dataclass(frozen=True)
class Share:
    """A share of the installed memory, as the settings file wrote it: 50%."""

    percent: int


@dataclass(frozen=True)
class Fitted:
    """Whatever is left once the weights that stay in system memory are counted."""


# How much system memory the router may hold processed prefixes in, as the person may
# state it. Absent from the settings file means Fitted: the machine works it out.
Budget = Fixed | Share | Fitted

# What a fitted budget leaves to everything that is not the router. Named rather than
# derived: what a desktop, a browser and a build need is not a property of this machine,
# and the models are meant to run with nobody logged in, where nothing else claims it.
SYSTEM_SHARE = Mib(8192)

GIBIBYTE = Mib(1024)


@dataclass(frozen=True)
class SystemMemory:
    """What the machine has, what the weights keep, and what is left for the cache."""

    installed: Mib
    resident: Mib
    cache: Mib


def system_memory(budget: Budget, machine: Machine, resident: Mib) -> SystemMemory:
    """How much of this machine's memory the prompt cache may hold.

    The cache competes with the model: `load-mode = mlock` asks for the part of the
    weights that did not go on the card to stay resident, and a cache sized past what is
    left pages those weights out -- generation collapses while prefill still looks
    healthy. A mixture of experts is where this bites, since the experts held in system
    memory are read for every token generated.

    So a fitted budget counts the heaviest such placement -- one model is resident at a
    time, so the heaviest is the one to leave room for -- and hands the system its share
    on top. A size or a percentage written down is taken as written: it is a judgement
    about this machine, and the settings file is where a person makes one.
    """
    match budget:
        case Fixed(size):
            cache = size
        case Share(percent):
            cache = Mib(round(machine.ram * percent / 100 / GIBIBYTE) * GIBIBYTE)
        case Fitted():
            left = machine.ram - resident - SYSTEM_SHARE
            cache = Mib(max(0, left) // GIBIBYTE * GIBIBYTE)

    return SystemMemory(installed=machine.ram, resident=resident, cache=cache)


def threads(cores: Sequence[Core]) -> int:
    """How many threads to run: the logical processors of the fastest cores.

    Not the physical cores and not all the logical ones. The pool is synchronised by a
    barrier, so every step runs at the speed of its slowest thread: fewer threads than
    the fast cores can carry gives up throughput, and more spreads the work onto slow
    cores that the rest then wait for.

    No affinity mask goes with this. Pinning a thread to a core keeps it there when the
    core is busy with something else, and the barrier waits for it anyway.
    """
    fastest = max(core.efficiency_class for core in cores)
    return sum(core.logical for core in cores if core.efficiency_class == fastest)


def parse_occupancy(text: str) -> NonEmpty[Occupancy]:
    """`nvidia-smi --query-gpu=memory.total,memory.free,name`, csv, no units.

    Every card, in the order nvidia-smi lists them. A placement uses the total and never
    the free: what is free is the state of a minute, and a preset outlives the minute it
    was written in. `vram` is about that minute and uses both.
    """
    read = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        total, free, name = fields
        if total.isdigit() and free.isdigit() and name:
            read.append(Occupancy(card=Card(name=name, total=Mib(int(total))),
                                  free=Mib(int(free))))

    match read:
        case []:
            raise UnreadableDevice(
                f"cannot read a card out of nvidia-smi: {text.strip()!r}")
        case [first, *rest]:
            return NonEmpty(first, *rest)


# What a placement needs to know of every card, as nvidia-smi is asked for it.
CARD_FIELDS = "index,name,memory.total,compute_cap,display_attached"

# How nvidia-smi says whether a monitor is plugged into a card.
_DISPLAY = {"Yes": True, "No": False}


def parse_cards(text: str) -> NonEmpty[Installed]:
    """`nvidia-smi --query-gpu=<CARD_FIELDS>`, csv, no units: every card in the machine.

    Every line has to read. A card passed over because its line did not is a card the
    placement never hears of, and a preset computed for a machine that does not exist.
    """
    read = [_installed(line) for line in text.splitlines() if line.strip()]

    match read:
        case []:
            raise UnreadableDevice(
                f"cannot read a card out of nvidia-smi: {text.strip()!r}")
        case [first, *rest]:
            return NonEmpty(first, *rest)


def _installed(line: str) -> Installed:
    """One card's line. A name is not split on: a card is named, not counted."""
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 5:
        raise UnreadableDevice(f"cannot read a card out of nvidia-smi: {line.strip()!r}")

    index, name, total, capability, display = fields
    major, _, minor = capability.partition(".")
    if not (index.isdigit() and name and total.isdigit() and major.isdigit()
            and minor.isdigit() and display in _DISPLAY):
        raise UnreadableDevice(f"cannot read a card out of nvidia-smi: {line.strip()!r}")

    return Installed(index=CudaIndex(int(index)),
                     card=Card(name=name, total=Mib(int(total))),
                     capability=Capability(int(major), int(minor)),
                     drives_display=_DISPLAY[display])
