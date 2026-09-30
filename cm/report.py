"""What calibrate says while it works.

Separate from the preset because it is written for a person rather than for the router:
it says what was decided and what was left out, in the order the models were worked
through. Nothing here is read back by anything.
"""

from collections.abc import Sequence
from pathlib import Path

from .machine import Card, SystemMemory
from .nonempty import NonEmpty
from .place import Endpoint, ExpertsOnCpu
from .render import Given, Offered, Placed
from .rpc import written
from .units import Mib


def opening(card: Card, reserve: Mib) -> str:
    """What the placements are being computed against: the whole of the card, since all
    of it is there to be placed on, and what is to be left free on it."""
    return f"{card.name}, {card.total} MiB, leaving about {reserve} free"


def remote(endpoint: Endpoint, available: Mib, reserve: Mib) -> str:
    """A slave's card, which is known here only by where it answers and what it was
    named as having."""
    return (f"slave at {written(endpoint)}, {available} MiB placeable, "
            f"leaving about {reserve} free")


def unreachable(endpoint: Endpoint) -> str:
    """A slave that is named and did not answer. The preset this run writes has no
    profile using its card, which is worth saying before anything else is."""
    return (f"The slave at {written(endpoint)} does not answer, so no profile uses its "
            "card this time. Start its worker there and run calibrate again: "
            "python -m cm.slave start")


def missing(key: str, path: Path) -> str:
    """A model whose file is not where its entry says.

    Nothing is placed for it and the run carries on: the other models are still to
    place. scan takes such an entry out, so seeing this means the file went between the
    two runs -- and the next scan is what tidies up after it.
    """
    return (f"{key}: no file at {path}, so nothing is placed for it. Fetch the file "
            "again, or take the entry out: python -m cm.scan")


def too_much(key: str, resident: Mib, room: Mib) -> str:
    """A model whose weights want more system memory than this machine has for them.

    Nothing is placed for it and the run carries on, the way a model with no file goes.
    Said with both numbers because the second one is a decision: a smaller `cache_ram`,
    or none written down at all, is more room for weights."""
    return (f"{key}: {resident} MiB of it would stay in system memory and this machine "
            f"has {room} MiB for weights, so nothing is placed for it.")


def system(memory: SystemMemory) -> str:
    """What the machine has to hold prefixes in, once the weights have had their share.

    Said out loud because it is the one number here that is not about the card, and the
    one a person is most likely to want to overrule -- a machine doing something else
    besides serving models wants less of it.
    """
    return (f"{memory.installed} MiB of system memory, {memory.resident} held by "
            f"weights off the card, {memory.cache} for cached prompts")


def no_projector(key: str, path: Path) -> str:
    """An entry naming a projector that is not there.

    Nothing is served for it and the run carries on, the way a model with no file goes.
    Said here rather than left to the load: the model would fail at the moment somebody
    asked for it, and this run is where the two files are looked at together.
    """
    return (f"{key}: no projector at {path}, so nothing is served for it. Fetch the file "
            "again, or take the projector out of the entry.")


def about(offered: Offered) -> tuple[str, ...]:
    """One model: its name, then a line per profile, or why there is none."""
    match offered:
        case Given():
            return (offered.model.key, f"  {_given(offered)}")
        case Placed() if not offered.profiles:
            return (offered.model.key,
                    "  nothing fits: it is served whole or not at all")
        case Placed():
            return (offered.model.key,
                    *(f"  {_profile(one)}"
                      for one in sorted(offered.profiles,
                                        key=lambda one: one.settings.ctx)))


def _given(given: Given) -> str:
    """A model nothing here placed: its one name, and what its entry says it holds.

    The window is not printed. It is in the entry's own settings, where a person wrote
    it, and repeating it here would read as something this run worked out.
    """
    return (f"{given.model.key:<26} as the settings file states it: "
            f"{given.stated.holds} MiB on the card")


def _profile(profile) -> str:
    settings = profile.settings
    offload = (f"  {settings.placement.layers} layers of experts on the cpu"
               if isinstance(settings.placement, ExpertsOnCpu) else "")

    return (f"{profile.name:<26} {settings.ctx:>7} tokens  "
            f"{settings.cache.value}  {_free(settings.spare)} MiB free{offload}")


def _free(spare: NonEmpty[Mib]) -> str:
    """What a placement leaves free: one figure for one device, one per device in order
    for several."""
    if len(spare) == 1:
        return f"{spare.first:>5}"
    return "/".join(str(one) for one in spare)


def closing(path: Path, offered: Sequence[Offered]) -> str:
    """What was written, and how much of it."""
    profiles = sum(_count(one) for one in offered)
    served = sum(1 for one in offered if _count(one))

    return f"{path}: {profiles} profiles from {served} of {len(offered)} models"


def _count(offered: Offered) -> int:
    """How many sections one model contributed."""
    match offered:
        case Given():
            return 1
        case Placed():
            return len(offered.profiles)
