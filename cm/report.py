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
from .render import Placed
from .rpc import written
from .units import Mib


def opening(card: Card, available: Mib, reserve: Mib) -> str:
    """What the placements are being computed against."""
    return (f"{card.name}, {card.total} MiB, {available} MiB placeable, "
            f"leaving about {reserve} free")


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
    """A model whose file is not where its entry says. Nothing is placed for it and the
    run carries on: an entry outlives the file it names -- scan never removes one -- and
    the other models are still to place."""
    return (f"{key}: no file at {path}, so nothing is placed for it. Fetch the file "
            "again, or take the entry out of the settings file.")


def system(memory: SystemMemory) -> str:
    """What the machine has to hold prefixes in, once the weights have had their share.

    Said out loud because it is the one number here that is not about the card, and the
    one a person is most likely to want to overrule -- a machine doing something else
    besides serving models wants less of it.
    """
    return (f"{memory.installed} MiB of system memory, {memory.resident} held by "
            f"weights off the card, {memory.cache} for cached prompts")


def about(placed: Placed) -> tuple[str, ...]:
    """One model: its name, then a line per profile, or why there is none."""
    if not placed.profiles:
        return (placed.model.key,
                "  nothing fits: it is served whole or not at all")

    lines = [placed.model.key]
    for profile in sorted(placed.profiles, key=lambda one: one.settings.ctx):
        lines.append(f"  {_profile(profile)}")

    return tuple(lines)


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


def closing(path: Path, placed: Sequence[Placed]) -> str:
    """What was written, and how much of it."""
    profiles = sum(len(one.profiles) for one in placed)
    served = sum(1 for one in placed if one.profiles)

    return f"{path}: {profiles} profiles from {served} of {len(placed)} models"
