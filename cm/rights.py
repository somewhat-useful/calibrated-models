"""Asking Windows for the rights a command needs, and running the command again with
them.

Two commands need rights this session may not have -- what the machine runs at startup
and what its firewall admits, both of which belong to the machine rather than to whoever
is logged in. What they do with those rights is entirely different; how they come by
them is the same thing twice, so it is here once.

Asked before anything is read or written, so that the elevated run is the one that does
the whole of the work rather than the second half of it. What comes back is what
happened: the elevated run and everything it wrote, this session being allowed to do the
work itself, or nothing having been done and the line saying why. The caller decides
what a refusal is called in its own words -- it is the one that knows what was not done.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from . import session, workspace


@dataclass(frozen=True)
class Held:
    """This session may do the work itself, so nothing was asked of anybody."""


@dataclass(frozen=True)
class Refused:
    """Nothing ran, and the one line saying why."""

    why: str


Rights = session.Ran | Held | Refused


@dataclass(frozen=True)
class Asking:
    """One command asking to be run again with the rights it needs.

    What it says it is changing goes into the sentence the person reads before the
    prompt comes up. A prompt that arrives without one is a prompt about nothing.
    """

    module: str
    said: tuple[str, ...]
    changing: str


def asked(asking: Asking) -> Rights:
    """The rights this run needs, asked for where this session does not have them."""
    if session.elevated():
        return Held()

    print(f"{asking.changing} needs an elevated session.")
    print("Asking Windows for one now -- answer the prompt.")
    print()

    match session.elevate(Path(sys.executable), ("-m", asking.module, *asking.said),
                          workspace.root()):
        case session.Ran(_, _) as ran:
            return ran
        case session.Declined():
            return Refused("The prompt was declined, so nothing was changed.")
        case session.Unavailable(why):
            return Refused(f"{why}.\n\nOpen PowerShell as administrator, and run:\n\n"
                           f"  cd {workspace.root()}\n"
                           f"  {_by_hand(asking)}")


def _by_hand(asking: Asking) -> str:
    """The same command, for somebody to type into an elevated session themselves."""
    return " ".join((sys.executable, "-m", asking.module, *asking.said))
