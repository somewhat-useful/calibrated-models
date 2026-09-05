"""The `vram` screen, as text.

Laid out here and drawn elsewhere, so that what it says can be checked without a
terminal. Nothing in these lines is an escape sequence: a line that is under the cursor
says so, and how a cursor looks is the drawing's business.

Two lists, read top to bottom. What the router can load and what each of those would
take, so that a model too big for the card as it stands says so here rather than failing
to allocate later. Then what is holding the card, largest first, with a box against each
one saying whether closing it is part of the plan. Every figure is read again on every
keystroke, because the point of the screen is to watch them move.

Every row is something a person can do. What holds the card and cannot be given up --
Windows' own machinery, the compositor, the window this is drawn in -- is a number under
the list and never rows. It is there so that the two lists account for the header: what
is in use is what is on the list, what the router hands back, and that number. The part
of it belonging to this window is said as well, because that part does come back -- once
the model is loaded and this window is closed.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NewType

from .advise import Fits, Holder, Pid, Room, Short, Unoffered, standing
from .catalog import Loadable
from .machine import Occupancy
from .units import Mib

# How many lines the console has room for.
Rows = NewType("Rows", int)

KEYS = ("  [tab] next model   [shift-tab] back   [space] mark   "
        "[enter] close marked   [r] refresh   [q] quit")

UP = "up"
DOWN = "down"
NEXT = "tab"
PREV = "shift-tab"
MARK = "space"

# Any other key: the screen is read again where it stands. That is the refresh.
STAY = ""


@dataclass(frozen=True)
class Line:
    """One line of the screen, and whether the cursor is on it."""

    text: str
    cursor: bool


def lines(occupancy: Occupancy, available: Mib, offered: Sequence[Loadable],
          chosen: Loadable, holders: Sequence[Holder], cursor: Pid,
          marked: frozenset[Pid], unoffered: Unoffered,
          room: Rows) -> tuple[Line, ...]:
    """The whole screen: the card, what can be loaded, what holds it, and the keys.

    Never longer than `room`, the lines the console has. A frame drawn taller than the
    window scrolls it, and the next frame drawn from the top lands in the middle of the
    last one -- which reads as a screen that has stopped answering.
    """
    head = [Line(_card(occupancy), False)]
    if available > occupancy.free:
        head.append(Line(_router(occupancy, available), False))

    head.append(Line("", False))
    for one in offered:
        head.append(Line(_model(one, available, one == chosen), False))
    head.append(Line("", False))

    tail = [Line("", False)]
    if unoffered.held > 0:
        tail.append(Line(_unoffered(unoffered.held), False))
        if unoffered.this_window > 0:
            tail.append(Line(_this_window(unoffered.this_window), False))
    for note in _notes(chosen, available, holders, marked):
        tail.append(Line(note, False))
    tail.append(Line(KEYS, False))

    shown = _fitting(holders, cursor, Rows(room - len(head) - len(tail)))
    body = [Line(_row(one, one.pid in marked), one.pid == cursor) for one in shown]
    if len(shown) < len(holders):
        body.append(Line(_rest(len(holders) - len(shown)), False))

    return tuple((head + body + tail)[:room])


def opens_on(available: Mib, offered: Sequence[Loadable]) -> Loadable:
    """Which profile the screen opens with picked.

    The first one that does not fit. A profile that already loads needs nothing closed,
    and opening on it would be a screen with nothing marked and nothing to do. Where
    every one of them loads, the largest: the first to stop fitting as the desktop grows.
    """
    if not offered:
        raise ValueError("a screen with nothing to load is not a screen")

    short = [one for one in offered if one.needs > available]
    return short[0] if short else max(offered, key=lambda one: one.needs)


def picked(chosen: Loadable, key: str, offered: Sequence[Loadable]) -> Loadable:
    """Which profile is picked after this key.

    NEXT and PREV walk the list and come back round, unlike the cursor: the profiles are
    a ring of things to choose between rather than a column to walk, and a person cycling
    it wants the one they passed rather than a stop at the end.
    """
    step = {NEXT: 1, PREV: -1}.get(key, 0)
    if step == 0 or not offered:
        return chosen
    if chosen not in offered:
        return offered[0]

    return offered[(offered.index(chosen) + step) % len(offered)]


def toggled(marked: frozenset[Pid], key: str, under: Holder) -> frozenset[Pid]:
    """The marked set after this key. MARK takes the row under the cursor in or out."""
    if key != MARK:
        return marked
    if under.pid in marked:
        return marked - {under.pid}

    return marked | {under.pid}


def _fitting(holders: Sequence[Holder], cursor: Pid, room: Rows) -> tuple[Holder, ...]:
    """As much of the list as there is room for, the cursor among it.

    Twenty-odd programs hold this card, which is more than a console window has lines
    for. What does not fit is counted rather than drawn, and the part drawn moves with
    the cursor so that walking the list is walking all of it.
    """
    if room <= 0 or not holders:
        return ()
    if len(holders) <= room:
        return tuple(holders)

    # One line of the room goes to saying how many are not on the screen.
    window = room - 1
    at = next((i for i, one in enumerate(holders) if one.pid == cursor), 0)
    first = min(max(0, at - window // 2), len(holders) - window)

    return tuple(holders[first:first + window])


def _card(occupancy: Occupancy) -> str:
    used = Mib(occupancy.card.total - occupancy.free)
    return (f"{occupancy.card.name}   {occupancy.card.total} MiB   "
            f"in use {used}   free {occupancy.free}")


def _router(occupancy: Occupancy, available: Mib) -> str:
    """Said out loud, because it is the one figure here nvidia-smi disagrees with."""
    held = Mib(available - occupancy.free)
    return (f"  the router holds {held} MiB and hands it back when it loads: "
            f"{available} to place in")


def _model(one: Loadable, available: Mib, chosen: bool) -> str:
    mark = ">" if chosen else " "
    return f"{mark} {one.name:<28} {one.needs:>6} MiB   {_standing(one, available)}"


def _standing(one: Loadable, available: Mib) -> str:
    match standing(Room(free=available, wanted=one.needs)):
        case Fits(spare):
            return f"loads now, {spare} MiB to spare"
        case Short(by):
            return f"short by {by} MiB"


def _row(holder: Holder, marked: bool) -> str:
    """A name, a number and a box. How the program is closed is not on the row: it is
    decided by what the program has on screen, not by anything a person chooses here,
    and a row saying it is a row asking to be read for nothing."""
    box = "[x]" if marked else "[ ]"

    return f"  {box} {holder.name:<28} {holder.pid:>7} {holder.held:>7} MiB"


def _unoffered(held: Mib) -> str:
    """The rest of what is in use, so that the screen accounts for its own header."""
    return (f"  the other {held} MiB is Windows, the compositor and this window "
            f"-- not on offer")


def _this_window(held: Mib) -> str:
    """The one part of that a person can still act on."""
    return f"  {held} MiB of that is this window, and comes back when it is closed"


def _rest(count: int) -> str:
    return f"      ... and {count} more holding less"


def _notes(chosen: Loadable, available: Mib, holders: Sequence[Holder],
           marked: frozenset[Pid]) -> tuple[str, ...]:
    """What is worth saying under the list, and nothing where there is nothing."""
    match standing(Room(free=available, wanted=chosen.needs)):
        case Fits(_):
            return ()
        case Short(by):
            return (_plan(by, holders, marked),)


def _plan(short_by: Mib, holders: Sequence[Holder], marked: frozenset[Pid]) -> str:
    """What closing the marked windows comes to, against what is missing."""
    if not [one for one in holders if one.held > 0]:
        return f"  {short_by} MiB to free, and nothing here that can be given up"

    chosen = [one for one in holders if one.pid in marked]
    if not chosen:
        return f"  nothing marked, and {short_by} MiB still to free"

    held = Mib(sum(one.held for one in chosen))
    counted = f"  marked {_programs(len(chosen))}, {held} MiB"
    if held >= short_by:
        return f"{counted} -- enough by these figures"

    # The marks do not add up to the shortfall, and would look wrong left unexplained.
    return f"{counted} of the {short_by} needed; the compositor gives back more on top"


def _programs(count: int) -> str:
    return f"{count} program" if count == 1 else f"{count} programs"


def moved(cursor: int, key: str, count: int) -> int:
    """Where the cursor goes next.

    It stops at the ends rather than wrapping: a list this short is read by looking at
    it, and a cursor that reappears at the other end is a cursor that got lost.
    """
    if count <= 0:
        return 0

    step = {UP: -1, DOWN: 1}.get(key, 0)
    return max(0, min(cursor + step, count - 1))
