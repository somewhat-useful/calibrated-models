"""What a profile is called, and therefore what a person asks the router for.

A name is built from the profile's own numbers, so the list the router serves says what
the choice between two profiles actually is. It says what a profile gives up and nothing
it keeps: a q8_0 cache and a prediction head that runs are how a model is meant to run,
and one card is the fastest way to run it, so none of them is written. What is written
is the window, a cache held otherwise, a head left out, and cards past the first -- each
is something a person choosing between two profiles pays for or is paid with.

That has a price, and it is worth stating. The numbers come from this card, so the same
model calibrated on a different card is served under a different name, and a client
holding qwen3.8-45k stops finding it. The alternative was to name a profile for what it
is for rather than what it is -- fast, long -- which survives a change of card and tells
the person nothing about what they get. This file takes the first.

A name is only ever written. Nothing reads a window, a cache or a head back out of one,
which is why the key it starts with may look like anything at all.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .place import CacheType, Layout, Settings, Variant, endpoints

# The k in a name is a thousand tokens. Windows are chosen a thousand at a time, so
# nothing is lost saying it that way; were that to change, the name would round and the
# profile would still run at the window it was given.
THOUSAND = 1000


@dataclass(frozen=True)
class Profile:
    """A placement together with the name the router serves it under."""

    name: str
    settings: Settings


def names(key: str, chosen: Sequence[Settings],
          ways: Sequence[Variant]) -> tuple[Profile, ...]:
    """Every profile of one model, named.

    The window is what tells profiles apart, so it is written only where there is
    something to tell apart: a model with a single profile is served without it, since
    gemma4-12b-262k would be noise on a model that has no other way to run. What a
    profile gives up is written however many there are.

    ways is every way the file may run here. A head left out is said only where running
    one was among them: a file carrying none, a head ruled out by hand, and a mixture,
    which never runs one, have given nothing up.
    """
    if len(chosen) == 1:
        return (Profile(f"{key}{_given_up(chosen[0], ways)}", chosen[0]),)

    return tuple(Profile(f"{key}-{settings.ctx // THOUSAND}k{_given_up(settings, ways)}",
                         settings)
                 for settings in chosen)


def _given_up(settings: Settings, ways: Sequence[Variant]) -> str:
    """The cache, the head and the cards, each where it is not the best there is."""
    return f"{_cache(settings.cache)}{_head(settings, ways)}{_cards(settings.layout)}"


def _cache(cache: CacheType) -> str:
    """q4_0 as q4: the width is the choice, the block layout comes with it.

    q8_0 is what a cache is held in unless a coarser one buys a window, so it goes
    unsaid. f16 is held only where the settings file asks for it, and is said.
    """
    match cache:
        case CacheType.Q8_0:
            return ""
        case CacheType.Q4_0:
            return "-q4"
        case CacheType.F16:
            return "-f16"


def _head(settings: Settings, ways: Sequence[Variant]) -> str:
    """-nomtp where the file could run a head here and this profile runs without it."""
    offered = any(one.head for one in ways)
    return "-nomtp" if offered and not settings.head else ""


def _cards(layout: Layout) -> str:
    """-2gpu for two of this machine's cards, -rpc where a slave's card is among them.

    A card reached over the network slows a profile more than any card in the machine,
    so -rpc says all there is and the cards beside it go unsaid.
    """
    if endpoints(layout):
        return "-rpc"
    if len(layout.devices) == 1:
        return ""
    return f"-{len(layout.devices)}gpu"
