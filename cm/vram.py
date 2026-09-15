"""vram: what the router can load right now, and what to close so it can load more.

The loop here decides nothing. It reads the preset once, the card and the session on
every pass, hands them to the screen, draws what comes back and waits for the next key.
The card is the one a desktop is drawn on: what holds any other is nothing a person
closes a window to give back.
What each profile costs was decided by calibrate and travels in the preset; what to give
up first is worked out in `advise`; where the memory belongs, and what the router is
holding on the next model's behalf, in `desktop`.

The wait for a key has a deadline, so a screen nobody is touching still reads the card
again and draws what it now says. That is what makes closing a window -- from here or
with the mouse -- something a person sees happen rather than something they have to ask
about.

The marks are a proposal, not a decision. They are made fresh for whichever profile is
picked, a person adds to them or takes from them, and nothing is closed until they say
so.
"""

import argparse
import ctypes
import os
import shutil
import sys
import time
from collections.abc import Sequence
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

from . import advise, catalog, desktop, devices, files, reading, screen, session
from .advise import Holder, Pid, Room, Unoffered
from .catalog import Loadable
from .config import ConfigError
from .machine import (Installed, NoDesktopCard, SeveralDesktopCards, UnreadableDevice,
                      desktop_card)
from .nonempty import NonEmpty
from .place import Local, device_name
from .screen import DOWN, MARK, NEXT, PREV, STAY, UP, Line, Rows
from .units import Mib


CLOSE = "enter"
QUIT = "q"

# How long the screen stands before it reads the card again with nothing pressed.
HEARTBEAT = 1.0

# How long a window is given to go after it has been asked to, and how often it is
# looked at while it goes.
PATIENCE = 5.0
GLANCE = 0.1

_HOME = "\x1b[H"
_TO_END_OF_LINE = "\x1b[K"
_TO_END_OF_SCREEN = "\x1b[J"
_REVERSE = "\x1b[7m"
_PLAIN = "\x1b[0m"

_STDOUT = -11
_ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004

_CONIN = "CONIN$"
_READ_WRITE = 0xC0000000
_SHARED = 0x00000003
_OPEN_EXISTING = 3

_SIGNALLED = 0x00000000
_KEY_EVENT = 0x0001
_SHIFT_PRESSED = 0x0010

_VK_TAB = 0x09

# Every key that is not a character. A key that is one arrives as itself.
_VK_NAMED = {0x0D: CLOSE, 0x20: MARK, 0x26: UP, 0x28: DOWN}

# Held down rather than pressed: shift, control, alt, the windows keys and the locks.
_VK_MODIFIERS = frozenset({0x10, 0x11, 0x12, 0x14, 0x5B, 0x5C, 0x90, 0x91})


class _Character(ctypes.Union):
    _fields_ = [("unicode", ctypes.c_wchar), ("ascii", ctypes.c_char)]


class _KeyEvent(ctypes.Structure):
    _fields_ = [("down", wintypes.BOOL), ("repeats", wintypes.WORD),
                ("vk", wintypes.WORD), ("scan", wintypes.WORD),
                ("character", _Character), ("state", wintypes.DWORD)]


class _Event(ctypes.Union):
    _fields_ = [("key", _KeyEvent)]


class _InputRecord(ctypes.Structure):
    _fields_ = [("kind", wintypes.WORD), ("event", _Event)]


def main(argv: Sequence[str] | None = None) -> int:
    given = _arguments(argv if argv is not None else sys.argv[1:])

    try:
        _watch(given.settings)
    except (ConfigError, UnreadableDevice) as refusal:
        print(refusal, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0

    return 0


def named(vk: int, char: str, shift: bool) -> str:
    """What a key event means to the screen.

    Taken from the key that was pressed and not from the character it produced, because
    the two keys that walk the list of profiles produce the same character: the console
    reports shift-tab as a tab with shift held.

    Anything not spoken for here is STAY, which draws the screen again where it stands.
    """
    if vk == _VK_TAB:
        return PREV if shift else NEXT
    if vk in _VK_NAMED:
        return _VK_NAMED[vk]
    if vk in _VK_MODIFIERS or not char.isprintable():
        return STAY

    return char.lower()


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vram",
        description="Show what the router can load on this card as it stands, and "
                    "close what is standing in the way of the rest.")
    parser.add_argument("--settings", type=Path, default=reading.DEFAULT,
                        help="the settings file naming the preset to read")

    return parser.parse_args(list(argv))


def _watch(settings: Path) -> None:
    read = reading.read(settings)
    preset = settings.parent / read.preset_path
    if not files.exists(preset):
        raise ConfigError(f"{preset.name} not found at {preset}; run calibrate")

    card = _watched(devices.cards())
    offered = catalog.parse(files.read(preset),
                            device_name(Local(card.index, card.card.total)))
    adapter = session.adapter(card.address)
    _accept_escapes()
    keyboard = _keyboard()
    ours = Pid(os.getpid())

    running = session.running(adapter)
    chosen = screen.opens_on(desktop.available(devices.occupancy(card.index).free, running),
                             offered)

    at = 0
    marked: frozenset[Pid] = frozenset()
    propose = True

    while True:
        running = session.running(adapter)
        occupancy = devices.occupancy(card.index)
        holders = desktop.holders(running, ours)
        available = desktop.available(occupancy.free, running)

        # Whatever was closed since the last draw shortens the list under the cursor and
        # leaves marks on windows that are gone, so both are put back on the list first.
        at = screen.moved(at, STAY, len(holders))
        marked = _still_open(marked, holders)
        if propose:
            marked = _proposed(available, chosen, holders)
            propose = False

        unoffered = Unoffered(
            held=desktop.not_offered(occupancy, available, holders),
            this_window=desktop.held_by_this_window(running, ours))

        _draw(screen.lines(occupancy, available, offered, chosen, holders,
                           _under(holders, at), marked, unoffered, _room()))

        key = _key(keyboard, HEARTBEAT)
        if key == QUIT:
            return

        if key == CLOSE:
            closing = desktop.closes(running, holders, marked)
            for pid in closing.asked:
                session.close(pid)
            for pid in closing.ended:
                session.end(pid)
            _gone(closing.asked | closing.ended, adapter)
            # What was asked for has been asked for. What that leaves is a new question,
            # and its answer is a fresh proposal rather than the old one minus.
            propose = True
        else:
            # Every other key redraws where it leaves the cursor, the marks and the
            # picked profile -- and so does no key at all, once the wait runs out.
            at = screen.moved(at, key, len(holders))
            if holders:
                marked = screen.toggled(marked, key, holders[at])

            picked = screen.picked(chosen, key, offered)
            propose = picked != chosen
            chosen = picked


def _watched(cards: NonEmpty[Installed]) -> Installed:
    """The card this screen is about, or the one line saying why there is none."""
    match desktop_card(cards):
        case Installed() as card:
            return card
        case NoDesktopCard():
            raise UnreadableDevice(
                f"None of this machine's {len(cards)} cards has a monitor plugged in, so "
                "no desktop is holding one that closing a window would give back.")
        case SeveralDesktopCards(count):
            raise UnreadableDevice(
                f"{count} cards on this machine have monitors plugged in, and vram "
                "watches the one card a desktop is drawn on.")


def _proposed(available: Mib, chosen: Loadable,
              holders: Sequence[Holder]) -> frozenset[Pid]:
    """What the screen opens with marked for this profile."""
    room = Room(free=available, wanted=chosen.needs)
    return frozenset(one.pid for one in advise.to_close(room, holders))


def _still_open(marked: frozenset[Pid], holders: Sequence[Holder]) -> frozenset[Pid]:
    """A mark on a program that has gone is a mark on whatever reuses its number."""
    return marked & {one.pid for one in holders}


def _under(holders: Sequence[Holder], at: int) -> Pid:
    """Whose row the cursor is on. No rows, no cursor: 0 is not a process id."""
    return holders[at].pid if holders else Pid(0)


def _gone(pids: frozenset[Pid], adapter: session.Adapter) -> None:
    """Wait for the programs that were asked to close to actually be gone.

    WM_CLOSE is a request and an application takes its time answering it, sometimes to
    ask whether the work should be saved. Drawing the moment after asking would draw the
    screen that was already there, which reads as nothing having happened. Waiting for
    ever would be worse, so an application that stops to ask holds this up only until
    PATIENCE runs out and then appears on the redrawn list, still open.
    """
    deadline = time.monotonic() + PATIENCE

    while pids and time.monotonic() < deadline:
        if not pids & {one.pid for one in session.running(adapter)}:
            return
        time.sleep(GLANCE)


def _room() -> Rows:
    """How many lines this console has, asked again every pass: windows get resized."""
    return Rows(shutil.get_terminal_size().lines)


def _draw(lines: Sequence[Line]) -> None:
    """Over the top of the last screen rather than after clearing it.

    The screen redraws itself on a heartbeat now, and clearing first makes that a flicker
    once a second. Each line wipes its own tail instead, and the last one wipes whatever
    a longer screen left below.

    Nothing is written after the last line, not even a newline: a newline on the bottom
    line of the window scrolls it, and the next frame would be drawn one line into the
    last one.
    """
    drawn = [(_REVERSE + one.text + _PLAIN if one.cursor else one.text)
             + _TO_END_OF_LINE
             for one in lines]

    sys.stdout.write(_HOME + "\n".join(drawn) + _TO_END_OF_SCREEN)
    sys.stdout.flush()


@dataclass(frozen=True)
class _Keyboard:
    """The console's input, and the library that reads it."""

    kernel32: ctypes.WinDLL
    handle: int


def _keyboard() -> _Keyboard:
    """The console's own input, opened by name.

    Not stdin: this program is watched while other things are done, and stdin may have
    been handed something else. CONIN$ is the keyboard of the console this is drawn on
    whatever stdin was pointed at.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                     ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.HANDLE]
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.ReadConsoleInputW.argtypes = [wintypes.HANDLE,
                                           ctypes.POINTER(_InputRecord),
                                           wintypes.DWORD,
                                           ctypes.POINTER(wintypes.DWORD)]

    handle = kernel32.CreateFileW(_CONIN, _READ_WRITE, _SHARED, None,
                                  _OPEN_EXISTING, 0, None)
    if handle == wintypes.HANDLE(-1).value:
        raise UnreadableDevice("this program is not running on a console it can read")

    return _Keyboard(kernel32=kernel32, handle=handle)


def _key(keyboard: _Keyboard, patience: float) -> str:
    """One keystroke, named -- or STAY once `patience` runs out with nothing pressed.

    Nothing pressed is an answer too, and the answer is to read the card again. Records
    that are not a key going down -- the mouse crossing the window, shift being held --
    are read past without spending the wait they arrived in.
    """
    record = _InputRecord()
    read = wintypes.DWORD()
    left = patience

    while left > 0:
        started = time.monotonic()
        if keyboard.kernel32.WaitForSingleObject(keyboard.handle,
                                                 int(left * 1000)) != _SIGNALLED:
            return STAY
        left -= time.monotonic() - started

        if not keyboard.kernel32.ReadConsoleInputW(keyboard.handle,
                                                   ctypes.byref(record), 1,
                                                   ctypes.byref(read)):
            return STAY
        if record.kind != _KEY_EVENT or not record.event.key.down:
            continue

        key = record.event.key
        spoken = named(key.vk, key.character.unicode,
                       bool(key.state & _SHIFT_PRESSED))
        if spoken != STAY:
            return spoken

    return STAY


def _accept_escapes() -> None:
    """Windows consoles ignore escape sequences until told not to."""
    if sys.platform != "win32":
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.GetStdHandle(_STDOUT)

    mode = ctypes.c_ulong()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return

    kernel32.SetConsoleMode(handle,
                            mode.value | _ENABLE_VIRTUAL_TERMINAL_PROCESSING)


if __name__ == "__main__":
    sys.exit(main())
