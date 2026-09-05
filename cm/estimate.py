"""What a configuration needs, read out of llama-fit-params' answer.

The estimator prints one line per device -- the device, then model, context and compute
in MiB. The card's line is what a placement has to fit into. The Host line is the other
half of the same placement: what stays in system memory, where it competes with the
prompt cache rather than with the card.

A refusal is its own answer rather than a zero. Zero is a configuration that needs no
memory, and every comparison downstream would wave it through.
"""

from dataclasses import dataclass

from .units import Mib

# The device holding what did not go on the card.
HOST = "Host"


@dataclass(frozen=True)
class Needs:
    """One placement, on both sides of the bus."""

    card: Mib
    host: Mib


@dataclass(frozen=True)
class Refused:
    """The estimator would not answer for this configuration.

    Deliberately carries no amount: there is no number here to be added to anything.
    """


Requirement = Needs | Refused


def parse_requirement(text: str) -> Requirement:
    rows = _rows(text)
    host = next((total for device, total in rows if device == HOST), Mib(0))

    for device, total in rows:
        if device != HOST:
            return Needs(card=total, host=host)

    return Refused()


def _rows(text: str) -> tuple[tuple[str, Mib], ...]:
    """Every device the answer speaks about, with its three amounts added up.

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
            read.append((fields[0], Mib(sum(int(amount) for amount in amounts))))

    return tuple(read)
