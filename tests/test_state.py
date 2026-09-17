"""Invariants of reading which layers keep a recurrent state, and what it costs.

A draft a prediction head makes can be rejected, and a recurrent state is rolled back from
snapshots the server keeps of it, one for every token drafted. What those cost follows
from two things the loader prints as it allocates the state: which layers keep one, and
what one sequence's comes to on each device. Read wrong, they misprice every head of
such a model without a word.
"""

import unittest
from fractions import Fraction

from cm.facts import MissingFact, Recurrent, Stateless, parse_facts
from test_facts import REQUIRED, verbose

KEPT = "0.00.521.121 D llama_memory_recurrent, layer {index:3}: dev = {device}"
SKIPPED = "0.00.521.125 D llama_memory_recurrent: layer {index:3}: skipped"
BUFFER = ("0.00.522.705 D llama_memory_recurrent:      {device} RS buffer size = "
          "{size:>8} MiB")


def layers(count, attending, device="CUDA0"):
    """A line for each layer, as the loader prints them: skipped where the layer attends."""
    return [SKIPPED.format(index=index) if index in attending
            else KEPT.format(index=index, device=device)
            for index in range(count)]


class AFileWithoutRecurrentLayersKeepsNoState(unittest.TestCase):
    def test_nothing_printed_is_nothing_kept(self):
        self.assertEqual(Stateless(), parse_facts(verbose(REQUIRED)).state)


class TheLayersGivenADeviceAreTheOnesThatKeepAState(unittest.TestCase):
    def test_a_skipped_layer_keeps_none(self):
        text = verbose(REQUIRED, [*layers(8, attending={3, 7}),
                                  BUFFER.format(device="CUDA0", size="18.70")])

        self.assertEqual(frozenset({0, 1, 2, 4, 5, 6}), parse_facts(text).state.layers)

    def test_each_keeps_an_equal_share_of_the_state(self):
        text = verbose(REQUIRED, [*layers(8, attending={3, 7}),
                                  BUFFER.format(device="CUDA0", size="18.70")])

        self.assertEqual(Fraction("18.70") / 6, parse_facts(text).state.mib_per_layer)

    def test_the_state_on_every_device_is_added_up(self):
        split = [*layers(4, attending={3}, device="CUDA1"),
                 *layers(8, attending={7})[4:],
                 BUFFER.format(device="CUDA0", size="9.35"),
                 BUFFER.format(device="CUDA1", size="9.35")]

        self.assertEqual(Recurrent(layers=frozenset({0, 1, 2, 4, 5, 6}),
                                   mib_per_layer=Fraction("18.70") / 6),
                         parse_facts(verbose(REQUIRED, split)).state)


class OneKindOfLineWithoutTheOtherIsReported(unittest.TestCase):
    def test_layers_without_what_their_state_costs(self):
        with self.assertRaises(MissingFact):
            parse_facts(verbose(REQUIRED, layers(8, attending={3, 7})))

    def test_a_cost_without_the_layers_that_keep_it(self):
        with self.assertRaises(MissingFact):
            parse_facts(verbose(REQUIRED, [BUFFER.format(device="CUDA0", size="18.70")]))


if __name__ == "__main__":
    unittest.main()
