"""What to close so that a model can be loaded.

The card is either free enough already, or a person has to give something up. Which of
the two it is, and what would have to go, is decided here and nowhere else: this module
reads nothing and kills nothing.

Proposed is only what pressing the key actually frees. A program with a window up is
asked to close, and one holding the card out of sight is ended, because asking a hidden
window does nothing.

What it proposes is taken from the big end, and a browser is taken before anything else.
Both are about what a person loses rather than about megabytes: the largest one frees
the most per thing given up, and a browser comes back with its tabs where they were,
which almost nothing else on a desktop does.

Nothing here ever answers "closing everything would still not be enough", because
nothing here could say that truthfully. What a program is credited with is a floor on
what closing it hands back and never a ceiling: on this machine the per-process figures
add up to 5098 MiB while the card holds 2482, and the difference is surfaces the
compositor is credited with on behalf of the windows drawing through it. Those come back
as well, by an amount nothing here can know in advance. So where the floor does not
reach, everything is proposed and the card is read again.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NewType

from .units import Mib

Pid = NewType("Pid", int)

# What a person loses least by closing. Named, because nothing Windows reports about a
# process says whether it is a browser, and there is no shape to recognise one by.
BROWSERS = frozenset({"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe",
                      "opera.exe", "opera_gx.exe", "vivaldi.exe", "librewolf.exe",
                      "zen.exe", "arc.exe", "yandex.exe"})


@dataclass(frozen=True)
class OnScreen:
    """A window is up. Asking it to close is what clicking its corner does."""


@dataclass(frozen=True)
class OutOfSight:
    """Holding the card with nothing on screen: in the notification area, or with no
    window at all. Hiding a window frees nothing -- measured, the window is still there
    at its full size with its surfaces on the card -- and asking a hidden window to
    close is not answered. So this one is ended rather than asked."""


# How a program is closed. Both of them are something a person can do; a program that
# is neither is not on the list at all.
Closing = OnScreen | OutOfSight


@dataclass(frozen=True)
class Holder:
    """A program on this desktop, and the video memory its processes hold."""

    name: str
    pid: Pid
    held: Mib
    closing: Closing


@dataclass(frozen=True, kw_only=True)
class Unoffered:
    """What is in use on the card that the list cannot offer: Windows' own machinery,
    the compositor, and the window this is drawn in.

    `this_window` is the part of `held` that comes back when that window is closed. It
    is the one thing in the figure a person can still act on, and it is said out loud
    because a screen that names a gigabyte and offers none of it invites the question
    of what the gigabyte is.

    Built by keyword: the second is part of the first, and the two swapped would say a
    console window holds a gigabyte.
    """

    held: Mib
    this_window: Mib


@dataclass(frozen=True)
class Room:
    """How much of the card is free, and how much of it a model wants."""

    free: Mib
    wanted: Mib


@dataclass(frozen=True)
class Fits:
    """What is free is already enough, with this much over."""

    spare: Mib


@dataclass(frozen=True)
class Short:
    """What is free falls this far short of what is wanted."""

    by: Mib


Standing = Fits | Short


def standing(room: Room) -> Standing:
    """Whether a model would fit as things are. Nothing about closing anything.

    Separate from the proposal because a screen listing many models asks this of every
    one of them, and asks what to close about only the one a person picked.
    """
    if room.free >= room.wanted:
        return Fits(Mib(room.free - room.wanted))

    return Short(Mib(room.wanted - room.free))


def to_close(room: Room, holders: Sequence[Holder]) -> tuple[Holder, ...]:
    """What to propose closing. Empty where the model already fits.

    Taken in the order above until the shortfall is covered, and no further: the last
    one taken is the one that covers it, so dropping it would leave the model short.
    """
    match standing(room):
        case Fits(_):
            return ()
        case Short(by):
            marked: list[Holder] = []
            freed = Mib(0)
            for one in order(holders):
                if freed >= by:
                    break
                marked.append(one)
                freed = Mib(freed + one.held)

            return tuple(marked)


def order(holders: Sequence[Holder]) -> tuple[Holder, ...]:
    """What to give up first: a browser, then whatever holds the most.

    A program holding nothing is left out entirely. Closing it frees nothing, and
    proposing it would be advice that cannot work.
    """
    return tuple(sorted((one for one in holders if one.held > 0),
                        key=lambda one: (one.name.lower() not in BROWSERS, -one.held)))
