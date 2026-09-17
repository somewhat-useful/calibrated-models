"""Who is holding the card, and what would give it back.

The process holding the memory is rarely the one a person would close: a program is
spread over processes, and the one drawing is a child of the one that was started. So
memory is gathered under the program it belongs to, and what the list names is programs.

What counts as one program is the executables belonging to the same piece of software:
the same file, or a different one from the same publisher. A program that draws its
interface in a helper of another name is one program still, and ending the helper on its
own is work the program undoes: it starts another.

Nothing here asks whether a program has a window before listing it. Most of what holds
this card is minimised to the notification area and has no window on screen at all;
leaving that out would leave out the answer to where the memory went.

How a program is closed is the other question each row answers. One with a window up
is asked, the way clicking its corner asks, and it can say that something is unsaved.
One holding the card out of sight is ended instead: hiding a window frees nothing and
asking a hidden window is not answered, so asking would be a key that does nothing.

Windows' own machinery is neither, and is not on the list either: a row that cannot be
acted on is a row to scroll past. What it holds is said in one line and no more. Windows
with a window up is another matter -- a folder window closes like any window, and that
is what closing it means: the window, never the shell behind it.

Three things are left off. Two because what they hold is nobody's to give back, and one
by identity:

    the compositor   its figure is the surfaces of the windows drawing through it,
                     counted on their behalf and given back with them
    the router       what it holds is already counted as available, because it unloads
                     one model before it loads the next
    this program     the terminal it is drawn in holds memory like anything else, and
                     closing it would end the program in the middle of being asked
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from .advise import Closing, Holder, OnScreen, OutOfSight, Pid
from .machine import Occupancy
from .units import Mib


@dataclass(frozen=True, kw_only=True)
class Running:
    """One process, as Windows reports it.

    Built by keyword: a process id and its parent's are the same kind of number, and
    the two of them positional would let a caller swap them with nothing to notice.

    `windowed` means a window up on the screen, the kind with a corner to click.
    `system` means the executable is Windows' own. `microsoft` is the wider of the two
    and holds wherever `system` does: the file says Microsoft wrote it, which is true of
    Windows' machinery living outside the Windows directory as well. `publisher` is what
    the file says about who wrote it, which is how a program is told from its helpers.
    """

    pid: Pid
    name: str
    parent: Pid
    held: Mib
    windowed: bool
    system: bool
    microsoft: bool
    publisher: str


# The compositor. Named, because nothing Windows reports about the process says that its
# figure is everybody else's.
COMPOSITOR = "dwm.exe"

# The router. Named rather than recognised by shape, because what singles it out is not
# anything Windows can see: it is the one process this program exists to make room for.
ROUTER = "llama-server.exe"


# What a card has to hold of the compositor's memory to be one a desktop is drawn on,
# as a share of what the card holding the most of it holds. The compositor holds memory
# on a card it draws no desktop on as well: a window rendered there and composed onto a
# screen elsewhere is copied across, and the copy is credited to the compositor. Those
# copies are a fraction of a desktop -- measured on this machine, 192 MiB against 2490
# on the card the monitors were plugged into -- while two cards each driving screens
# hold the same order of it, in proportion to the screens.
DESKTOP_SHARE = Fraction(1, 4)


def desktops(held: Sequence[Mib]) -> tuple[bool, ...]:
    """Which of these cards a desktop is drawn on, from what the compositor holds on each.

    The compositor draws the desktop, so it holds the most where the desktop is, and
    that card is one. So is any card holding within DESKTOP_SHARE of it: monitors on two
    cards are two desktops, and both want room to work at.

    Nothing is asked about monitors: a remote session detaches this machine's, and the
    desktop goes on costing the card what it costs. A machine nobody is logged in to has
    no compositor holding anything anywhere, and no card a desktop is drawn on.
    """
    most = max(held, default=Mib(0))
    if most <= 0:
        return tuple(False for _ in held)

    return tuple(one >= most * DESKTOP_SHARE for one in held)


def available(free: Mib, running: Sequence[Running]) -> Mib:
    """What a model may have: the card's free memory, and what the router already holds.

    The router keeps one model resident and unloads it before it loads another, so its
    memory is the next model's memory. Counting it as taken would report a model that is
    loaded right now as one that does not fit.
    """
    serving = sum(one.held for one in running if one.name == ROUTER)
    return Mib(free + serving)


def holders(running: Sequence[Running], ours: Pid) -> tuple[Holder, ...]:
    """The programs holding video memory that can be given up, largest first.

    Largest first because the list is read to answer "what do I give up", and the answer
    is at that end. A column that has to be read to the bottom before it says anything
    is a column sorted for the wrong question.

    Only what can be given up. Windows' own machinery holds a third of this card and
    cannot be closed by anybody; listing it would be twenty rows to scroll past to reach
    the two that matter. `not_offered` is what that comes to, for the line it deserves.

    `ours` is this program's own process. The program it is running in is left off the
    list: what it holds is real, but it cannot be offered by the thing that would stop
    running the moment it was accepted.
    """
    by_pid = {one.pid: one for one in running}

    listed = [Holder(by_pid[pid].name, pid, Mib(sum(one.held for one in family)), how)
              for pid, family in _programs(running, ours).items()
              for how in [_closing(family)]
              if how is not None and sum(one.held for one in family) > 0]

    return tuple(sorted(listed, key=lambda one: (-one.held, one.name, one.pid)))


def not_offered(occupancy: Occupancy, available: Mib,
                listed: Sequence[Holder]) -> Mib:
    """What the card says is in use that the list does not offer: Windows' own
    machinery, the compositor, and the window this is drawn in.

    By subtraction, and never by adding processes up. The per-process figures overlap:
    the compositor is credited with the surfaces of the windows drawing through it, and
    each of those surfaces is counted again under the program it belongs to. Measured on
    this machine: 1388 MiB across the processes against 1041 in use on the card. A figure
    added up that way disagrees with the header above it, and a screen whose own numbers
    disagree is a screen checked against nvidia-smi rather than read.

    Subtraction agrees by construction. What is in use is three things and nothing else:
    what the list can give back, what the router hands back when it loads -- already
    counted into `available` -- and this.

    Never below nothing. The card and the processes are read a moment apart and from
    different places, so the list can name memory the card has already been given back.
    """
    offered = sum(one.held for one in listed)

    return Mib(max(0, occupancy.card.total - available - offered))


def _programs(running: Sequence[Running],
              ours: Pid) -> dict[Pid, list[Running]]:
    """Every process gathered under the program it belongs to, ours and the two that
    hold on nobody's behalf left out."""
    by_pid = {one.pid: one for one in running}
    ourselves = _ours(ours, by_pid)

    gathered: dict[Pid, list[Running]] = {}
    for one in running:
        gathered.setdefault(_program(one, by_pid).pid, []).append(one)

    return {pid: family for pid, family in gathered.items()
            if pid not in ourselves
            and not any(one.name in (COMPOSITOR, ROUTER) for one in family)}


def _closing(family: Sequence[Running]) -> Closing | None:
    """How this program is closed, or None where it is Windows' own machinery.

    A window up decides it, whoever wrote the program: a folder window, a browser, an
    editor and the task manager are all closed the same way, and closing one is closing
    the window rather than the program behind it.

    With nothing on screen there is nothing to ask, and what happens next depends on
    whose the program is. The person's is ended -- that is what a person goes to the
    task manager for. Windows' own is left alone: its machinery is started again the
    moment Windows wants it, and ending it is work the machine undoes. That is measured
    by two things the file says about itself and by where it lives, never by its name.
    """
    if any(one.windowed for one in family):
        return OnScreen()
    if any(one.system or one.microsoft for one in family):
        return None

    return OutOfSight()


def family(running: Sequence[Running], program: Pid) -> tuple[Pid, ...]:
    """Every process of one program, the one it is named for first.

    Closing a program means reaching its windows, and the windows are not always the
    named process's. Measured on this machine: Notepad is two processes, the memory and
    the window both in the second, and the first is the one the list names. Asking only
    that one asks nobody.

    Named first because ending a program ends this list in order, and a program whose
    first process is still running starts its helpers again.
    """
    by_pid = {one.pid: one for one in running}
    mine = tuple(one.pid for one in running if _program(one, by_pid).pid == program)

    return tuple(sorted(mine, key=lambda pid: pid != program))


@dataclass(frozen=True, kw_only=True)
class Closes:
    """What [enter] does to the marked programs.

    Two sets and not one, because the two are not the same act: asking is a message a
    program may answer or refuse, ending is not. Built by keyword so that the two,
    being the same kind of thing, cannot be swapped by a caller.
    """

    asked: frozenset[Pid]
    ended: frozenset[Pid]


def closes(running: Sequence[Running], holders: Sequence[Holder],
           marked: frozenset[Pid]) -> Closes:
    """Which processes to ask to close, and which to end outright.

    Both are whole programs: a program is reached through all of its processes, whether
    it is being asked or ended.
    """
    asked: set[Pid] = set()
    ended: set[Pid] = set()

    for one in holders:
        if one.pid not in marked:
            continue
        match one.closing:
            case OnScreen():
                asked.update(family(running, one.pid))
            case OutOfSight():
                ended.update(family(running, one.pid))

    return Closes(asked=frozenset(asked), ended=frozenset(ended))


def _program(one: Running, by_pid: Mapping[Pid, Running]) -> Running:
    """Which process stands for this one: the topmost of the same piece of software.

    A program spreads over processes and the memory is held by the ones a person never
    sees. What holds them together is the software they are part of, so the walk goes up
    while the parent runs the same executable or one from the same publisher, and stops
    where neither holds. Above that point is whoever started the program -- the shell, a
    service host, a terminal -- and never part of it: walking on regardless would credit
    half the desktop to explorer.

    Windows' own is where every walk stops. It launches most of what runs here, and it
    is the publisher of a good deal of it as well.
    """
    seen: set[Pid] = set()

    while one.pid not in seen:
        seen.add(one.pid)

        # A process id is reused once its process is gone, so a parent chain can close
        # on itself or point at something that was never the parent. Both end here.
        parent = by_pid.get(one.parent)
        if parent is None or parent.system or not _same_software(parent, one):
            return one
        one = parent

    return one


def _same_software(parent: Running, one: Running) -> bool:
    """Whether these two processes are parts of one program rather than one program
    having started another.

    The same file is the plain case: a browser draws each tab in a copy of itself. The
    same publisher is the other one: a program whose interface is a separate executable,
    written by whoever wrote the one above it. A file that says nothing about its
    publisher claims nobody as kin.
    """
    return (parent.name == one.name
            or bool(one.publisher) and parent.publisher == one.publisher)


def held_by_this_window(running: Sequence[Running], ours: Pid) -> Mib:
    """What goes with the window this is drawn in, when a person closes it.

    Said under the list because it is the one part of what the list cannot offer that a
    person can still act on: the console is holding video memory like anything else, and
    it hands it back on the way out. It is not a row, because a row is something to mark
    and marking this one would end the program in the middle of being asked.
    """
    by_pid = {one.pid: one for one in running}
    mine = _ours(ours, by_pid)

    return Mib(sum(one.held for one in running
                   if _program(one, by_pid).pid in mine))


def _ours(ours: Pid, by_pid: Mapping[Pid, Running]) -> frozenset[Pid]:
    """The programs that go when this one does: this program itself, and the windows it
    is running in, since closing any of them closes it."""
    mine = {_program(by_pid[pid], by_pid).pid for pid in _running_us(ours, by_pid)}
    if ours in by_pid:
        mine.add(_program(by_pid[ours], by_pid).pid)

    return frozenset(mine)


def _running_us(ours: Pid, by_pid: Mapping[Pid, Running]) -> frozenset[Pid]:
    """The windows this program is running in: closing any of them closes it.

    Every window above it, not only the nearest. A terminal inside an editor is closed
    by closing either of the two, so both of them are the program's own window as far as
    this question goes.
    """
    found: set[Pid] = set()
    seen: set[Pid] = set()

    one = by_pid.get(ours)
    while one is not None and one.pid not in seen:
        seen.add(one.pid)
        if one.windowed:
            found.add(one.pid)
        one = by_pid.get(one.parent)

    return frozenset(found)
