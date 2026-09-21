"""Asking this machine what it is. No decision is taken here.

Windows only, and it says so rather than guessing. The processor topology and the
installed memory come from the operating system through ctypes; there is no portable
way to read either, and a wrong answer would be a placement computed for a machine that
does not exist.
"""

import ctypes
import struct
import sys
from ctypes import wintypes
from collections.abc import Sequence
from pathlib import Path

from . import desktop, lmstudio, session
from .lmstudio import Found, Library, Missing
from .machine import (CARD_FIELDS, Attached, Core, CudaIndex, Installed, Machine,
                      Occupancy, UnreadableDevice, parse_cards, parse_occupancy)
from .nonempty import NonEmpty
from .proc import run
from .units import Mib
from .upstream import Cuda, Driver, supported

# GetLogicalProcessorInformationEx, RelationProcessorCore: one record per physical core.
RELATION_PROCESSOR_CORE = 0

# Layout of what it returns, on any 64-bit Windows. SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX
# is a DWORD relationship and a DWORD size, then the union; PROCESSOR_RELATIONSHIP is a
# flags byte, an efficiency class byte, twenty reserved bytes, a group count, and then
# one GROUP_AFFINITY per group -- a mask whose set bits are this core's threads.
_RELATIONSHIP = 8
_EFFICIENCY_CLASS = _RELATIONSHIP + 1
_GROUP_COUNT = _RELATIONSHIP + 22
_FIRST_GROUP = _RELATIONSHIP + 24
_GROUP_SIZE = 16


def probe() -> Machine:
    """This machine: its cards, its memory, and its cores."""
    if sys.platform != "win32":
        raise UnreadableDevice(
            f"reading this machine is implemented for Windows only, not {sys.platform}")

    return Machine(cards=cards(), ram=_ram(), cores=_cores())


def library() -> Library:
    """Where LM Studio keeps its weights on this machine.

    Not behind the Windows gate above: LM Studio records its home the same way on every
    system it runs on, and a machine that has none answers Missing rather than failing.
    """
    profile = Path.home()
    home = lmstudio.home(profile, _text(profile / lmstudio.POINTER))
    root = lmstudio.models(home, _text(home / lmstudio.SETTINGS))

    return Found(root) if root.is_dir() else Missing(root)


def _text(path: Path) -> str:
    """A file's text, or "" where there is no file to read.

    Both files are somebody else's, kept for their own purposes. One that is absent,
    locked or unreadable is one that does not name a library, which is a fact rather
    than a failure.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def cards() -> NonEmpty[Installed]:
    """Every card in this machine, as a placement needs to know it.

    Two readings. The card itself comes from nvidia-smi; whether a desktop draws on it
    comes from Windows' own counters, which say where the compositor holds memory. The
    monitors are nobody's answer to that: a remote session detaches them, and the driver
    then reports none on any card while the desktop holds what it held.
    """
    cards_here = attached()
    first, *rest = (Installed(index=one.index, card=one.card, capability=one.capability,
                              draws_desktop=drawn, address=one.address)
                    for one, drawn in zip(cards_here, _desktops(cards_here)))
    return NonEmpty(first, *rest)


def attached() -> NonEmpty[Attached]:
    """Every card in this machine, as the driver describes it."""
    done = run(("nvidia-smi", f"--query-gpu={CARD_FIELDS}",
                "--format=csv,noheader,nounits"))
    if not done.out.strip():
        raise UnreadableDevice(f"nvidia-smi said nothing: {done.err.strip()!r}")

    return parse_cards(done.out)


def _desktops(attached: Sequence[Attached]) -> tuple[bool, ...]:
    """Which of these cards a desktop is drawn on.

    Every card at once, because the counters name every adapter in one reading and what
    counts as a desktop is a card's share of them. A card Windows lists no display
    adapter for draws no desktop: a desktop is drawn with an adapter, and it has none.
    """
    adapters = tuple(session.adapter(one.address) for one in attached)
    known = tuple(one for one in adapters if isinstance(one, session.Adapter))
    if not known:
        return tuple(False for _ in adapters)

    drawn = dict(zip(known, desktop.desktops(session.held_by(desktop.COMPOSITOR, known))))
    return tuple(drawn.get(one, False) for one in adapters)


def occupancy(index: CudaIndex) -> Occupancy:
    """One card, and how much of it is free at this moment."""
    done = run(("nvidia-smi", f"--id={index}",
                "--query-gpu=memory.total,memory.free,name",
                "--format=csv,noheader,nounits"))
    if not done.out.strip():
        raise UnreadableDevice(f"nvidia-smi said nothing: {done.err.strip()!r}")

    return parse_occupancy(done.out).first


def _ram() -> Mib:
    """Installed memory, as GlobalMemoryStatusEx reports it.

    A little under the sticks in the machine: what the firmware holds back never becomes
    memory anything can allocate, and half of what cannot be allocated is not a cache.
    """

    class Status(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD),
                    ("memory_load", wintypes.DWORD),
                    ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong),
                    ("avail_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended_virtual", ctypes.c_ulonglong)]

    status = Status()
    status.length = ctypes.sizeof(Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise UnreadableDevice("GlobalMemoryStatusEx would not say how much memory this "
                               "machine has")

    return Mib(status.total_phys // 1048576)


# How long a driver version NVML writes can be, terminator included: its own
# NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE.
_DRIVER_VERSION_SIZE = 80


def driver() -> Driver:
    """The NVIDIA driver here: the version it goes by, and the newest CUDA it runs."""
    return Driver(version=_driver_version(), cuda=cuda_driver())


def _driver_version() -> str:
    """The driver's version, as NVML reports it.

    Asked of NVML rather than of nvidia-smi, whose driver_version field says it is
    deprecated in favour of one older drivers do not have.
    """
    if sys.platform != "win32":
        raise UnreadableDevice(
            f"reading this machine is implemented for Windows only, not {sys.platform}")

    try:
        nvml = ctypes.WinDLL("nvml.dll")
    except OSError as absent:
        raise UnreadableDevice("there is no NVIDIA driver here to say which version it "
                               f"is: {absent}") from None

    status = nvml.nvmlInit_v2()
    if status != 0:
        raise UnreadableDevice(f"NVML would not start: it answered {status}")

    try:
        written = ctypes.create_string_buffer(_DRIVER_VERSION_SIZE)
        status = nvml.nvmlSystemGetDriverVersion(written, _DRIVER_VERSION_SIZE)
        if status != 0:
            raise UnreadableDevice("NVML would not say which version the driver is: it "
                                   f"answered {status}")
        return written.value.decode("ascii", "replace")
    finally:
        nvml.nvmlShutdown()


def cuda_driver() -> Cuda:
    """The newest CUDA version this machine's driver runs, as the driver itself says.

    Asked of the driver library through cuDriverGetVersion rather than read off
    nvidia-smi: nvidia-smi prints the same figure only in its report, under a label it
    has already renamed once and says it will drop in CUDA 14.
    """
    if sys.platform != "win32":
        raise UnreadableDevice(
            f"reading this machine is implemented for Windows only, not {sys.platform}")

    try:
        driver = ctypes.WinDLL("nvcuda.dll")
    except OSError as absent:
        raise UnreadableDevice("there is no NVIDIA driver here to say which CUDA version "
                               f"it runs: {absent}") from None

    encoded = ctypes.c_int()
    status = driver.cuDriverGetVersion(ctypes.byref(encoded))
    if status != 0:
        raise UnreadableDevice(f"cuDriverGetVersion would not say which CUDA version the "
                               f"driver runs: it answered {status}")

    return supported(encoded.value)


def _cores() -> tuple[Core, ...]:
    raw = _topology()

    cores = []
    offset = 0
    while offset < len(raw):
        size = struct.unpack_from("<I", raw, offset + 4)[0]
        if size == 0:
            raise UnreadableDevice("the processor topology does not parse")

        efficiency = raw[offset + _EFFICIENCY_CLASS]
        groups = struct.unpack_from("<H", raw, offset + _GROUP_COUNT)[0]
        logical = sum(
            struct.unpack_from("<Q", raw,
                               offset + _FIRST_GROUP + group * _GROUP_SIZE)[0].bit_count()
            for group in range(groups))

        cores.append(Core(efficiency_class=efficiency, logical=logical))
        offset += size

    if not cores:
        raise UnreadableDevice("this machine reports no processor cores")

    return tuple(cores)


def _topology() -> bytes:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    length = wintypes.DWORD(0)

    kernel32.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, None,
                                              ctypes.byref(length))
    buffer = (ctypes.c_byte * length.value)()
    if not kernel32.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buffer,
                                                     ctypes.byref(length)):
        raise UnreadableDevice(
            f"GetLogicalProcessorInformationEx failed: {ctypes.get_last_error()}")

    return bytes(buffer)
