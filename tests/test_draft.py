"""Invariants of what a prediction head costs beyond what the estimator counts.

The estimator cannot be told a head will run. The server then holds on the last device
the head's weights, its cache and the working buffers of the draft context it runs; and
on every device a snapshot of each recurrent layer's state for every token the head
drafts. The numbers here are arbitrary; what is fixed is where each cost lands.
"""

import math
import unittest
from dataclasses import replace
from fractions import Fraction

from cm import place
from cm.estimate import Needs
from cm.facts import Head, ModelFacts, Recurrent
from cm.machine import CudaIndex
from cm.nonempty import NonEmpty
from cm.place import (Among, CacheType, Endpoint, Layout, Local, Pipeline, Remote,
                      Variant)
from cm.units import Halvings, Layers, Mib, Port, Tokens
from test_devices import DENSE_WITH_HEAD, law, local, run, two

ATTENDING = frozenset(range(3, 64, 4))
STATE = Recurrent(layers=frozenset(range(64)) - ATTENDING, mib_per_layer=Fraction("3.5"))

PLAIN = ModelFacts(n_expert=0, n_layer=Layers(64), n_ctx_train=Tokens(262144),
                   head=Head(weight_bytes=350_000_000, cache_per_token=2048))
HYBRID = replace(PLAIN, state=STATE)

ANSWER = Needs(cards=NonEmpty(Mib(4000), Mib(10000)),
               working=NonEmpty(Mib(300), Mib(700)), host=Mib(500))

# Blocks 0-22 on the first card and 23-63 on the second, the output and the head's own
# block on the second as well.
LAYOUT = Layout(devices=NonEmpty(Local(CudaIndex(1), Mib(8192)),
                                 Local(CudaIndex(0), Mib(16303))),
                layers=NonEmpty(Layers(23), Layers(43)), halvings=Halvings(0),
                pipeline=Pipeline.ON, among=Among.SEVERAL)

DRAFTING = Variant(CacheType.Q8_0, head=True)
UNDRAFTED = Variant(CacheType.Q8_0, head=False)
CTX = Tokens(32000)


class TheLastDeviceCarriesTheHeadAndItsDraft(unittest.TestCase):
    def test_without_a_head_it_is_what_the_estimator_said_and_the_context(self):
        """The estimator knows nothing of the context CUDA holds on every device it runs
        on, and a placement that counted less than a profile holds would be wrong."""
        self.assertEqual(NonEmpty(*(Mib(one + place.CUDA_CONTEXT) for one in ANSWER.cards)),
                         place.requirement(HYBRID, UNDRAFTED, CTX, ANSWER,
                                           LAYOUT.devices))

    def test_with_one_the_last_device_adds_the_head_and_the_drafts_working_buffers(self):
        needed = place.requirement(HYBRID, DRAFTING, CTX, ANSWER, LAYOUT.devices)

        self.assertEqual(ANSWER.cards.first + place.CUDA_CONTEXT, needed.first)
        self.assertEqual(ANSWER.cards.last + place.head_cost(HYBRID, CacheType.Q8_0, CTX)
                         + ANSWER.working.last + place.CUDA_CONTEXT, needed.last)


class ASlavesCardIsNotCountedTheContextOfThisMachine(unittest.TestCase):
    """What a slave keeps on its card is one figure a person wrote into the settings
    file, and the worker's own context is inside it."""

    LENT = Layout(devices=NonEmpty(Remote(Endpoint("worker", Port(50052)), Mib(12288)),
                                   Local(CudaIndex(0), Mib(16303))),
                  layers=LAYOUT.layers, halvings=LAYOUT.halvings,
                  pipeline=LAYOUT.pipeline, among=LAYOUT.among)

    def test_the_slaves_card_is_counted_what_the_estimator_said(self):
        needed = place.requirement(HYBRID, UNDRAFTED, CTX, ANSWER, self.LENT.devices)

        self.assertEqual(ANSWER.cards.first, needed.first)

    def test_this_machines_card_is_counted_the_context_besides(self):
        needed = place.requirement(HYBRID, UNDRAFTED, CTX, ANSWER, self.LENT.devices)

        self.assertEqual(ANSWER.cards.last + place.CUDA_CONTEXT, needed.last)


class EveryDeviceKeepsSnapshotsOfItsRecurrentLayers(unittest.TestCase):
    def test_one_per_token_drafted_of_every_layer_on_it_that_keeps_a_state(self):
        each = place.DRAFT_LOOKAHEAD * Fraction("3.5")

        self.assertEqual(NonEmpty(Mib(math.ceil(each * 18)), Mib(math.ceil(each * 30))),
                         place.snapshots(HYBRID, DRAFTING, LAYOUT))

    def test_none_without_a_head(self):
        self.assertEqual(NonEmpty(Mib(0), Mib(0)),
                         place.snapshots(HYBRID, UNDRAFTED, LAYOUT))

    def test_none_where_no_layer_keeps_a_state(self):
        self.assertEqual(NonEmpty(Mib(0), Mib(0)), place.snapshots(PLAIN, DRAFTING, LAYOUT))

    def test_a_placement_pays_for_them_in_window(self):
        hybrid = replace(DENSE_WITH_HEAD,
                         state=Recurrent(layers=frozenset(range(64)) - ATTENDING,
                                         mib_per_layer=Fraction(40)))
        chains = local(two())

        stateless, _ = run(DENSE_WITH_HEAD, chains, law(DENSE_WITH_HEAD))
        stateful, _ = run(hybrid, chains, law(DENSE_WITH_HEAD))

        def longest(chosen):
            return {tuple(one.layout.devices): one.ctx for one in chosen}

        shorter = longest(stateful)
        self.assertTrue(shorter)
        for devices, ctx in shorter.items():
            with self.subTest(devices=devices):
                self.assertLess(ctx, longest(stateless)[devices])


if __name__ == "__main__":
    unittest.main()
