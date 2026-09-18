"""What a profile is called, and therefore what a person asks the router for.

A name is built from the profile's own numbers, so the list the router serves says what
the choice between two profiles actually is: how long a window, how exactly the
conversation is held, whether a prediction head runs, whether it needs another machine's
card. That has a price, and it is worth stating. The numbers come from this card, so the
same model calibrated on a different card is served under a different name, and a client
holding qwen3.8-45k-q8 stops finding it. The alternative was to name a profile for what
it is for rather than what it is -- fast, long -- which survives a change of card and
tells the person nothing about what they get. This file takes the first.

A name is only ever written. Nothing reads a window, a cache or a head back out of one,
which is why the key it starts with may look like anything at all.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .place import CacheType, Settings, endpoints

# The k in a name is a thousand tokens. Windows are chosen a thousand at a time, so
# nothing is lost saying it that way; were that to change, the name would round and the
# profile would still run at the window it was given.
THOUSAND = 1000


@dataclass(frozen=True)
class Profile:
    """A placement together with the name the router serves it under."""

    name: str
    settings: Settings


def names(key: str, chosen: Sequence[Settings]) -> tuple[Profile, ...]:
    """Every profile of one model, named.

    Naming is a property of the set rather than of one profile: a model with a single
    profile is served under its key alone, because there is nothing to tell it apart
    from and gemma4-12b-262k-q8 would be noise on a model that has no other way to run.

    Needing a slave's card is not a way of telling profiles apart, so it is said however
    many there are: such a profile loads only while that machine's worker runs.
    """
    if len(chosen) == 1:
        return (Profile(f"{key}{_lent(chosen[0])}", chosen[0]),)

    return tuple(Profile(f"{key}-{_apart(settings)}{_lent(settings)}", settings)
                 for settings in chosen)


def _apart(settings: Settings) -> str:
    """What tells one profile of a model from the others: its window, cache and head.

    Nothing about how the layers are shared out appears. A dense model's layers are all
    on the cards, and a mixture has one profile and never reaches here, so the count of
    offloaded experts would distinguish nothing.
    """
    head = "-mtp" if settings.head else ""
    return f"{settings.ctx // THOUSAND}k-{_cache(settings.cache)}{head}"


def _lent(settings: Settings) -> str:
    """-rpc where the profile uses a slave's card, last in the name."""
    return "-rpc" if endpoints(settings.layout) else ""


def _cache(cache: CacheType) -> str:
    """q8_0 as q8: the width is the choice, the block layout comes with it."""
    return cache.value.split("_")[0]
