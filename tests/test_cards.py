"""Invariants of reading every card in the machine out of nvidia-smi.

The lines are written the way the driver prints them with noheader and nounits, for the
fields the placement asks for. Nothing here runs nvidia-smi.
"""

import unittest

from cm.machine import (CARD_FIELDS, Capability, Card, CudaIndex, Installed,
                        UnreadableDevice, parse_cards)
from cm.units import Mib

TWO = ("0, NVIDIA GeForce RTX 5070 Ti, 16303, 12.0, Yes\n"
       "1, NVIDIA GeForce RTX 2070, 8192, 7.5, No\n")


class EveryCardIsRead(unittest.TestCase):
    def test_both_cards_come_back_in_the_order_printed(self):
        self.assertEqual(
            [Installed(CudaIndex(0), Card("NVIDIA GeForce RTX 5070 Ti", Mib(16303)),
                       Capability(12, 0), drives_display=True),
             Installed(CudaIndex(1), Card("NVIDIA GeForce RTX 2070", Mib(8192)),
                       Capability(7, 5), drives_display=False)],
            list(parse_cards(TWO)))

    def test_one_card_is_a_list_of_one(self):
        self.assertEqual(1, len(parse_cards(TWO.splitlines()[0])))

    def test_blank_lines_are_not_cards(self):
        self.assertEqual(2, len(parse_cards("\n" + TWO + "\n\n")))

    def test_the_fields_asked_for_are_the_fields_read(self):
        """Five of them, in this order: a field added to the question and not to the
        reading would shift every one after it."""
        self.assertEqual(["index", "name", "memory.total", "compute_cap",
                          "display_attached"], CARD_FIELDS.split(","))


class TheGenerationOrdersAsANumber(unittest.TestCase):
    def test_twelve_is_later_than_seven_point_five(self):
        """As text, "12.0" sorts before "7.5"."""
        self.assertLess(Capability(7, 5), Capability(12, 0))

    def test_the_minor_number_breaks_a_tie(self):
        self.assertLess(Capability(8, 6), Capability(8, 9))


class ALineThatDoesNotReadIsRefused(unittest.TestCase):
    """A card left out because its line did not read is a card the placement never hears
    of, so one such line refuses the whole reading."""

    def test_a_field_the_driver_could_not_give(self):
        for line in ("0, NVIDIA GeForce RTX 5070 Ti, 16303, 12.0, [N/A]",
                     "0, NVIDIA GeForce RTX 5070 Ti, [N/A], 12.0, Yes",
                     "0, NVIDIA GeForce RTX 5070 Ti, 16303, [N/A], Yes"):
            with self.subTest(line=line):
                with self.assertRaises(UnreadableDevice) as refusal:
                    parse_cards(TWO + line + "\n")

                self.assertIn(line, str(refusal.exception))

    def test_a_name_with_a_comma_in_it_does_not_become_a_field(self):
        with self.assertRaises(UnreadableDevice):
            parse_cards("0, Some, Card, 16303, 12.0, Yes\n")

    def test_a_driver_that_says_nothing_is_refused(self):
        for text in ("", "\n", "No devices were found\n"):
            with self.subTest(text=text):
                with self.assertRaises(UnreadableDevice):
                    parse_cards(text)


if __name__ == "__main__":
    unittest.main()
