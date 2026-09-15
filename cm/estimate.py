"""What a configuration needs, read out of llama-fit-params' answer.

The estimator prints one line per device -- the device, then model, context and compute
in MiB. The card lines are what a placement has to fit into, one per device and in the
order the devices were given. The Host line is the other half of the same placement:
what stays in system memory, where it competes with the prompt cache rather than with a
card.

The compute column is kept apart as well as added in. A prediction head runs a draft
context of its own, and its working buffers are no larger than the model's own on the
same device -- which the estimator counts, although it is never asked about the draft.

A refusal is its own answer rather than a zero. Zero is a configuration that needs no
memory, and every comparison downstream would wave it through.
"""

from dataclasses import dataclass

from .nonempty import NonEmpty
from .units import Mib

# The device holding what did not go on a card.
HOST = "Host"


@dataclass(frozen=True)
class Needs:
    """One placement, on every device it uses and on the host.

    cards is everything each device holds, and working the part of it each holds as
    working buffers.
    """

    cards: NonEmpty[Mib]
    working: NonEmpty[Mib]
    host: Mib


@dataclass(frozen=True)
class Refused:
    """The estimator would not answer for this configuration.

    Deliberately carries no amount: there is no number here to be added to anything.
    """


Requirement = Needs | Refused


@dataclass(frozen=True)
class _Row:
    """One device the answer speaks about: all it holds, and its working buffers."""

    device: str
    total: Mib
    working: Mib


def parse_requirement(text: str) -> Requirement:
    rows = _rows(text)
    host = next((row.total for row in rows if row.device == HOST), Mib(0))

    match tuple(row for row in rows if row.device != HOST):
        case ():
            return Refused()
        case (first, *rest):
            return Needs(cards=NonEmpty(first.total, *(one.total for one in rest)),
                         working=NonEmpty(first.working, *(one.working for one in rest)),
                         host=host)


def _rows(text: str) -> tuple[_Row, ...]:
    """Every device the answer speaks about, with its three amounts.

    A device name is followed by exactly those three numbers. Anything else on a line --
    a log message, a partial answer -- is not one of these lines.
    """
    read = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue

        amounts = fields[1:4]
        if all(amount.isdigit() for amount in amounts):
            model, context, compute = (int(amount) for amount in amounts)
            read.append(_Row(device=fields[0], total=Mib(model + context + compute),
                             working=Mib(compute)))

    return tuple(read)
