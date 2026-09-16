"""What this machine is, in the numbers a placement and a preset need.

Nothing here reads a device. The figures arrive from devices.py already measured, and
what is computed from them is computed the same way whoever asks.
"""

import re
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
class PciAddress:
    """Where a card sits on the PCI bus. Windows names a card by an identifier nvidia-smi
    never prints, and this is what the two of them agree on."""

    bus: int
    device: int
    function: int


@dataclass(frozen=True)
class Attached:
    """A card in this machine, as the driver describes it.

    Everything about a card that is the card's own. What the machine does with it --
    whether a desktop draws on it -- is read somewhere else and joins it in Installed.
    """

    index: CudaIndex
    card: Card
    capability: Capability
    address: PciAddress


@dataclass(frozen=True)
class Installed:
    """A card in this machine, and what placing a model across several of them needs.

    draws_desktop is whether the desktop draws on this card, which is a fact about where
    its memory goes rather than about cabling: a card drawing a desktop is one somebody
    may be working at, and a placement leaves it room for that. Over a remote session
    this machine's monitors are detached and no card has one, while the desktop goes on
    holding what it holds, so it is not a monitor that is asked about.
    """

    index: CudaIndex
    card: Card
    capability: Capability
    draws_desktop: bool
    address: PciAddress


@dataclass(frozen=True)
class NoDesktopCard:
    """Of this machine's several cards, not one draws a desktop."""


@dataclass(frozen=True)
class SeveralDesktopCards:
    """More than one of this machine's cards draws a desktop."""

    count: int


def desktop_card(cards: NonEmpty[Installed]
                 ) -> Installed | NoDesktopCard | SeveralDesktopCards:
    """The card a desktop is drawn on: the one somebody closes windows to make room on.

    A machine with one card has that card, desktop or not. Of several, it is the one the
    desktop draws on; what holds the others is nothing a person can close.
    """
    if len(cards) == 1:
        return cards.first

    match [one for one in cards if one.draws_desktop]:
        case []:
            return NoDesktopCard()
        case [one]:
            return one
        case [_, _, *_] as showing:
            return SeveralDesktopCards(len(showing))


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
            cache = _share(percent, machine)
        case Fitted():
            left = machine.ram - resident - SYSTEM_SHARE
            cache = Mib(max(0, left) // GIBIBYTE * GIBIBYTE)

    return SystemMemory(installed=machine.ram, resident=resident, cache=cache)


def _share(percent: int, machine: Machine) -> Mib:
    """A share of the installed memory, rounded to whole gibibytes."""
    return Mib(round(machine.ram * percent / 100 / GIBIBYTE) * GIBIBYTE)


def off_card(budget: Budget, machine: Machine) -> Mib:
    """How much of this machine's memory the weights that leave the cards may take.

    A mixture of experts is placed by moving experts into system memory, and there is
    only so much of that: weights counted on but not held are read off the disk for
    every token generated, or fail to be locked at all where the settings file asks for
    them to be. So the cards are not the only bound a placement has to be inside, and
    this is the other one.

    What the machine has for weights is what it has beyond the system's own share, less
    a cache the settings file asked for by name -- that size is a judgement already
    made, and the weights come after it. A fitted cache takes nothing here: it is what
    is left once the weights have taken theirs, which is the same rule read from the
    other end.
    """
    match budget:
        case Fixed(size):
            asked = size
        case Share(percent):
            asked = _share(percent, machine)
        case Fitted():
            asked = Mib(0)

    return Mib(max(0, machine.ram - SYSTEM_SHARE - asked))


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


# What a placement needs to know of every card, as nvidia-smi is asked for it. Whether a
# desktop draws on a card is not among them: nvidia-smi answers that with the monitors,
# which a remote session detaches, and the desktop draws where it draws either way.
CARD_FIELDS = "index,name,memory.total,compute_cap,pci.bus_id"

# How nvidia-smi writes a card's place on the bus: the domain, the bus and the device,
# then the function, every one of them in hexadecimal.
_BUS_ID = re.compile(r"^[0-9A-Fa-f]{4,8}:([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.([0-9A-Fa-f])$")


def parse_cards(text: str) -> NonEmpty[Attached]:
    """`nvidia-smi --query-gpu=<CARD_FIELDS>`, csv, no units: every card in the machine.

    Every line has to read. A card passed over because its line did not is a card the
    placement never hears of, and a preset computed for a machine that does not exist.
    """
    read = [_attached(line) for line in text.splitlines() if line.strip()]

    match read:
        case []:
            raise UnreadableDevice(
                f"cannot read a card out of nvidia-smi: {text.strip()!r}")
        case [first, *rest]:
            return NonEmpty(first, *rest)


def _attached(line: str) -> Attached:
    """One card's line. A name is not split on: a card is named, not counted."""
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 5:
        raise UnreadableDevice(f"cannot read a card out of nvidia-smi: {line.strip()!r}")

    index, name, total, capability, bus_id = fields
    major, _, minor = capability.partition(".")
    address = _BUS_ID.match(bus_id)
    if not (index.isdigit() and name and total.isdigit() and major.isdigit()
            and minor.isdigit() and address is not None):
        raise UnreadableDevice(f"cannot read a card out of nvidia-smi: {line.strip()!r}")

    bus, device, function = (int(part, 16) for part in address.groups())
    return Attached(index=CudaIndex(int(index)),
                    card=Card(name=name, total=Mib(int(total))),
                    capability=Capability(int(major), int(minor)),
                    address=PciAddress(bus=bus, device=device, function=function))
