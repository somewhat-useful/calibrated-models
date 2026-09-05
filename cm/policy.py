"""The rule pi's context-policy extension applies, so pi's own numbers can stay behind it.

pi compacts a conversation on two global numbers: how much of the window it holds back
for the reply, and how much of the recent conversation it keeps word for word. One pair
of numbers cannot suit a fleet whose windows run from 28k to 262k, so the extension
computes both per model, from the window and the reply cap that model is listed with.

An extension cannot switch pi's own compaction off, only postpone it. So the policy
governs a model only while pi's numbers sit behind the ones it computed: at or above its
reserve, pi cuts first and the extension stands aside; keeping more than it would keep,
pi decides there is nothing worth compacting and never asks. What has to hold, then, is
that pi's pair fits behind the smallest pair the policy will produce for anything this
router serves.

The constants and the arithmetic below are the extension's own, read from its index.ts.
Nothing here is a judgement of ours and there is nothing to tune: it is a copy of an
external rule, and it is wrong in the way any copy is wrong once the original moves.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .served import Served
from .units import Tokens

OUTPUT_HEADROOM = 1.1
MIN_RESERVE = Tokens(4096)
RESERVE_SHARE = 0.3
RESERVE_FLOOR = Tokens(2048)
KEEP_SHARE = 0.35
MIN_KEEP = Tokens(2000)
MAX_KEEP = Tokens(48000)

# What pi's own numbers are rounded down to. They are read off a screen and compared with
# the policy's by hand, so they are worth having round.
PAGE = Tokens(500)


@dataclass(frozen=True)
class Policy:
    """What the extension holds back for the reply, and keeps verbatim, for one model."""

    reserve: Tokens
    keep: Tokens


@dataclass(frozen=True)
class Unpoliced:
    """A window the extension declines.

    With the reserve taken out of it there is no tail left worth keeping, so it computes
    nothing and the model is left to pi's own compaction.
    """

    window: Tokens


Decided = Policy | Unpoliced


def decided(model: Served) -> Decided:
    """What the extension will compute for one model, once pi is serving it.

    The extension's fallback for a model that declares no reply cap is not mirrored: every
    model this program writes is written with one.
    """
    reserve = min(max(_rounded(model.cap * OUTPUT_HEADROOM), MIN_RESERVE),
                  max(math.floor(model.window * RESERVE_SHARE), RESERVE_FLOOR))

    threshold = model.window - reserve
    if threshold <= MIN_KEEP:
        return Unpoliced(model.window)

    return Policy(reserve=Tokens(reserve),
                  keep=Tokens(min(max(math.floor(threshold * KEEP_SHARE), MIN_KEEP),
                                  MAX_KEEP)))


@dataclass(frozen=True)
class Bound:
    """The largest value pi may hold, and what to write in place of one above it."""

    most: Tokens
    take: Tokens


@dataclass(frozen=True)
class Room:
    """What pi's two numbers have to fit in for the policy to govern every model served."""

    reserve: Bound
    keep: Bound


@dataclass(frozen=True)
class Unbounded:
    """Nothing served is policed, so nothing here has an opinion about pi's numbers."""


def room(models: Sequence[Served]) -> Room | Unbounded:
    """The room the policy leaves pi across everything the router serves.

    Decided by whichever model produces the smallest numbers -- usually the one with the
    smallest reply cap rather than the smallest window, since the cap drives the reserve
    until the window gets small enough for the share of it to bite.
    """
    policed = []
    for model in models:
        match decided(model):
            case Policy() as one:
                policed.append(one)
            case Unpoliced():
                pass

    if not policed:
        return Unbounded()

    # Behind the reserve means strictly under it, and the reserve is never less than
    # RESERVE_FLOOR, so what is taken here stays well clear of zero.
    reserve = Tokens(min(one.reserve for one in policed) - 1)
    keep = Tokens(min(one.keep for one in policed))
    take = _page(reserve)

    return Room(reserve=Bound(most=reserve, take=take),
                keep=Bound(most=keep, take=min(_page(Tokens(take // 2)), _page(keep))))


def _rounded(count: float) -> int:
    """A token count as the extension rounds one: halves away from zero, not to even."""
    return math.floor(count + 0.5)


def _page(count: Tokens) -> Tokens:
    return Tokens(count // PAGE * PAGE)
