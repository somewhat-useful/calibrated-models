"""One card, as the placement sees it, for tests about a machine that has one.

Everything a placement is about is a list with an element per device, and a machine with
one card is the list with one element in it. These are those lists, written once rather
than in every test that needs one.
"""

from cm.estimate import Needs
from cm.machine import Capability, Card, CudaIndex, Installed
from cm.nonempty import NonEmpty
from cm.place import Among, Chain, Layout, Local, Pipeline, local_seat
from cm.units import Halvings, Layers, Mib

# The card these tests are written for, and the number llama.cpp gives it.
INDEX = CudaIndex(0)
TOTAL = Mib(16303)

# The micro-batch the settings file runs with, as it ships.
UBATCH = 512

# Every layer of a model on the one card.
LAYOUT = Layout(devices=NonEmpty(Local(INDEX, TOTAL)),
                layers=NonEmpty(Layers(65)),
                halvings=Halvings(0),
                pipeline=Pipeline.OFF,
                among=Among.ONE)


def installed(card: Card) -> NonEmpty[Installed]:
    """The cards of a machine with this one card in it."""
    return NonEmpty(Installed(index=INDEX, card=card, capability=Capability(12, 0),
                              drives_display=True))


def chains(total: Mib, reserve: Mib) -> NonEmpty[Chain]:
    """The one chain a machine with one card of this size places on."""
    return NonEmpty(NonEmpty(local_seat(INDEX, total, reserve)))


def needs(card: Mib, host: Mib) -> Needs:
    """An answer from the estimator about one card."""
    return Needs(cards=NonEmpty(card), host=host)
