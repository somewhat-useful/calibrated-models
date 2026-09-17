"""Invariants of reading what a configuration needs out of llama-fit-params' answer.

The estimator prints one line per device -- device, then model, context and compute in
MiB. The card's line is what the placement has to fit into; the Host line is the rest of
the same placement, held in system memory, where it competes with the prompt cache
rather than with the card. When the estimator will not answer at all, that is not zero:
zero would be a configuration needing no memory, and every comparison downstream would
accept it.
"""

import unittest

from cm.estimate import Needs, Refused, parse_requirement
from cm.nonempty import NonEmpty
from cm.units import Mib


def card(model: int, context: int, working: int) -> Needs:
    """A placement that keeps nothing in system memory."""
    return Needs(cards=NonEmpty(Mib(model + context + working)),
                 working=NonEmpty(Mib(working)), host=Mib(0))


class TheRequirementIsAllThreeNumbers(unittest.TestCase):
    def test_sums_model_context_and_compute(self):
        self.assertEqual(parse_requirement("CUDA0 12726 285 509\n"),
                         card(12726, 285, 509))

    def test_no_single_number_is_mistaken_for_the_answer(self):
        """Distinct numbers: a parser returning any one of them gives a different sum."""
        answer = parse_requirement("CUDA0 12726 285 509\n")

        self.assertNotIn(answer, (card(12726, 0, 0), card(0, 285, 0), card(0, 0, 509)))

    def test_trailing_space_does_not_change_the_answer(self):
        """The real output ends each line with a space."""
        self.assertEqual(parse_requirement("CUDA0 12726 285 509 \n"),
                         card(12726, 285, 509))


class TheWorkingBuffersAreTheLastNumber(unittest.TestCase):
    """A prediction head's draft context holds working buffers of its own, and they are
    measured against the model's on the same device."""

    def test_each_device_keeps_its_own(self):
        text = "CUDA1 3830 388 371 \nCUDA0 8896 823 711 \n"

        self.assertEqual(NonEmpty(Mib(371), Mib(711)), parse_requirement(text).working)

    def test_they_are_still_part_of_what_the_device_holds(self):
        answer = parse_requirement("CUDA0 100 20 30\n")

        self.assertEqual(NonEmpty(Mib(150)), answer.cards)
        self.assertEqual(NonEmpty(Mib(30)), answer.working)


class TheHostIsNotTheCard(unittest.TestCase):
    def test_both_sides_of_one_placement_are_read(self):
        """The real answer for a model kept partly in system memory."""
        text = "CUDA0 12726 285 509 \nHost 520 0 24\n"

        self.assertEqual(parse_requirement(text),
                         Needs(cards=NonEmpty(Mib(13520)), working=NonEmpty(Mib(509)),
                               host=Mib(544)))

    def test_the_host_is_never_added_to_the_card(self):
        with_host = parse_requirement("CUDA0 100 20 30\nHost 4000 0 0\n")

        self.assertEqual(with_host.cards, card(100, 20, 30).cards)

    def test_the_order_of_the_lines_does_not_matter(self):
        """The estimator prints the host first for some models and last for others."""
        first = parse_requirement("Host 520 0 24\nCUDA0 12726 285 509\n")
        last = parse_requirement("CUDA0 12726 285 509\nHost 520 0 24\n")

        self.assertEqual(first, last)

    def test_a_placement_with_no_host_line_holds_nothing_off_the_card(self):
        self.assertEqual(parse_requirement("CUDA0 100 20 30\n").host, Mib(0))

    def test_host_alone_is_a_refusal(self):
        """Nothing was placed on a card, so there is no requirement to report."""
        self.assertEqual(parse_requirement("Host 520 0 24\n"), Refused())


class TheDeviceMayBeCalledAnything(unittest.TestCase):
    def test_the_name_is_not_checked_against_a_list(self):
        """Another build names its device differently, and that is not a refusal."""
        for name in ("CUDA0", "ROCm0", "Vulkan1", "SYCL0"):
            with self.subTest(device=name):
                self.assertEqual(parse_requirement(f"{name} 100 20 30\n"),
                                 card(100, 20, 30))

    def test_every_device_line_is_an_answer_in_the_order_printed(self):
        text = "CUDA0 100 20 30\nCUDA1 900 90 9\n"

        self.assertEqual(parse_requirement(text),
                         Needs(cards=NonEmpty(Mib(150), Mib(999)),
                               working=NonEmpty(Mib(30), Mib(9)), host=Mib(0)))


class ARefusalIsNotANumber(unittest.TestCase):
    def test_no_device_line_at_all(self):
        text = "0.00.070.681 I llama_fit_params: printing estimated memory in MiB\n"

        self.assertEqual(parse_requirement(text), Refused())

    def test_empty_output(self):
        self.assertEqual(parse_requirement(""), Refused())

    def test_fewer_than_three_numbers(self):
        self.assertEqual(parse_requirement("CUDA0 6637 544\n"), Refused())

    def test_a_refusal_carries_no_amount(self):
        """Refused has nothing to add up, so no code path can mistake it for a size."""
        refused = parse_requirement("")

        self.assertFalse(hasattr(refused, "cards"))
        self.assertFalse(hasattr(refused, "host"))


if __name__ == "__main__":
    unittest.main()
