"""Invariants of reading every card in the machine out of nvidia-smi.

The lines are written the way the driver prints them with noheader and nounits, for the
fields the placement asks for. Nothing here runs nvidia-smi.
"""

import unittest

from cm.machine import (CARD_FIELDS, Capability, Card, CudaIndex, Installed, NoDesktopCard,
                        PciAddress, SeveralDesktopCards, UnreadableDevice, desktop_card,
                        parse_cards)
from cm.nonempty import NonEmpty
from cm.units import Mib

TWO = ("0, NVIDIA GeForce RTX 5070 Ti, 16303, 12.0, Yes, 00000000:01:00.0\n"
       "1, NVIDIA GeForce RTX 2070, 8192, 7.5, No, 00000000:05:00.0\n")


def installed(index, display) -> Installed:
    return Installed(CudaIndex(index), Card(f"card {index}", Mib(8192)), Capability(8, 6),
                     drives_display=display, address=PciAddress(index + 1, 0, 0))


class EveryCardIsRead(unittest.TestCase):
    def test_both_cards_come_back_in_the_order_printed(self):
        self.assertEqual(
            [Installed(CudaIndex(0), Card("NVIDIA GeForce RTX 5070 Ti", Mib(16303)),
                       Capability(12, 0), drives_display=True,
                       address=PciAddress(bus=1, device=0, function=0)),
             Installed(CudaIndex(1), Card("NVIDIA GeForce RTX 2070", Mib(8192)),
                       Capability(7, 5), drives_display=False,
                       address=PciAddress(bus=5, device=0, function=0))],
            list(parse_cards(TWO)))

    def test_one_card_is_a_list_of_one(self):
        self.assertEqual(1, len(parse_cards(TWO.splitlines()[0])))

    def test_blank_lines_are_not_cards(self):
        self.assertEqual(2, len(parse_cards("\n" + TWO + "\n\n")))

    def test_the_fields_asked_for_are_the_fields_read(self):
        """Six of them, in this order: a field added to the question and not to the
        reading would shift every one after it."""
        self.assertEqual(["index", "name", "memory.total", "compute_cap",
                          "display_attached", "pci.bus_id"], CARD_FIELDS.split(","))

    def test_the_place_on_the_bus_is_read_in_hexadecimal(self):
        line = "0, NVIDIA GeForce RTX 2070, 8192, 7.5, No, 00000000:0A:1F.3\n"

        self.assertEqual(PciAddress(bus=10, device=31, function=3),
                         parse_cards(line).first.address)


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
        for line in ("0, NVIDIA GeForce RTX 5070 Ti, 16303, 12.0, [N/A], 00000000:01:00.0",
                     "0, NVIDIA GeForce RTX 5070 Ti, [N/A], 12.0, Yes, 00000000:01:00.0",
                     "0, NVIDIA GeForce RTX 5070 Ti, 16303, [N/A], Yes, 00000000:01:00.0",
                     "0, NVIDIA GeForce RTX 5070 Ti, 16303, 12.0, Yes, [N/A]"):
            with self.subTest(line=line):
                with self.assertRaises(UnreadableDevice) as refusal:
                    parse_cards(TWO + line + "\n")

                self.assertIn(line, str(refusal.exception))

    def test_a_name_with_a_comma_in_it_does_not_become_a_field(self):
        with self.assertRaises(UnreadableDevice):
            parse_cards("0, Some, Card, 16303, 12.0, Yes, 00000000:01:00.0\n")

    def test_a_driver_that_says_nothing_is_refused(self):
        for text in ("", "\n", "No devices were found\n"):
            with self.subTest(text=text):
                with self.assertRaises(UnreadableDevice):
                    parse_cards(text)


class TheDesktopIsDrawnOnTheCardWithAMonitor(unittest.TestCase):
    def test_a_machines_only_card_is_its_desktop_card_monitor_or_not(self):
        for display in (True, False):
            only = installed(0, display)
            with self.subTest(display=display):
                self.assertEqual(only, desktop_card(NonEmpty(only)))

    def test_of_several_it_is_the_one_with_a_monitor(self):
        showing = installed(1, True)

        self.assertEqual(showing,
                         desktop_card(NonEmpty(installed(0, False), showing,
                                               installed(2, False))))

    def test_of_several_with_none_there_is_none(self):
        self.assertEqual(NoDesktopCard(),
                         desktop_card(NonEmpty(installed(0, False), installed(1, False))))

    def test_of_several_with_more_than_one_it_says_how_many(self):
        self.assertEqual(SeveralDesktopCards(2),
                         desktop_card(NonEmpty(installed(0, True), installed(1, False),
                                               installed(2, True))))


if __name__ == "__main__":
    unittest.main()
