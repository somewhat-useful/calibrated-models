"""Invariants of the numbers taken off this machine.

Nothing here reads a device. The measurements are written into the test, because what
is being checked is the arithmetic on them and an arithmetic that only holds for one
processor is not an arithmetic.
"""

import unittest

from cm.machine import (SYSTEM_SHARE, Card, Core, Fitted, Fixed, Machine,
                        Share, SystemMemory, UnreadableDevice, off_card,
                        parse_occupancy, system_memory, threads)
from cm.units import Mib
from one_card import installed

# What the two machines the reference file was written for report.
DESKTOP = tuple([Core(1, 2)] * 8 + [Core(0, 1)] * 8)      # i7-13700F, 8P + 8E
LAPTOP = tuple([Core(1, 2)] * 8 + [Core(0, 1)] * 16)      # i9-13950HX, 8P + 16E
NO_HYBRID = tuple([Core(0, 2)] * 6                        # one class, hyperthreaded
                  )


def machine(ram: int) -> Machine:
    """A machine of a given size. Nothing here depends on its card."""
    return Machine(cards=installed(Card("test", Mib(16303))), ram=Mib(ram), cores=DESKTOP)


THIS_ONE = machine(65407)

SIZES = range(4096, 262145, 331)


class ASizeWrittenDownIsTakenAsWritten(unittest.TestCase):
    """It is a judgement about the machine, and the settings file is where one is made."""

    def test_the_machine_does_not_argue_with_it(self):
        for ram in SIZES:
            with self.subTest(ram=ram):
                given = system_memory(Fixed(Mib(32768)), machine(ram), Mib(0))

                self.assertEqual(Mib(32768), given.cache)

    def test_not_even_when_the_weights_would_not_leave_room(self):
        given = system_memory(Fixed(Mib(60000)), THIS_ONE, Mib(20000))

        self.assertEqual(Mib(60000), given.cache)


class AShareIsOfWhatTheMachineHas(unittest.TestCase):
    def test_half_is_half_to_within_the_rounding(self):
        for ram in SIZES:
            with self.subTest(ram=ram):
                given = system_memory(Share(50), machine(ram), Mib(0))

                self.assertLessEqual(abs(given.cache * 2 - ram), 1024)

    def test_it_is_a_whole_number_of_gibibytes(self):
        for percent in (1, 33, 50, 75, 100):
            for ram in SIZES:
                with self.subTest(percent=percent, ram=ram):
                    given = system_memory(Share(percent), machine(ram), Mib(0))

                    self.assertEqual(given.cache % 1024, 0)

    def test_more_memory_never_holds_less(self):
        given = [system_memory(Share(50), machine(ram), Mib(0)).cache for ram in SIZES]

        self.assertEqual(given, sorted(given))

    def test_the_weights_do_not_enter_it(self):
        """A share is what was asked for, not what is left."""
        asked = system_memory(Share(50), THIS_ONE, Mib(0)).cache

        self.assertEqual(asked, system_memory(Share(50), THIS_ONE, Mib(20000)).cache)


class WhatIsLeftIsWhatTheWeightsLeave(unittest.TestCase):
    """mlock holds the weights that did not fit on the card; a cache sized past what is
    left pages them out, and generation collapses while prefill still looks healthy."""

    def test_the_weights_and_the_system_both_get_out_of_the_way(self):
        given = system_memory(Fitted(), THIS_ONE, Mib(12000))

        self.assertLessEqual(given.cache + Mib(12000) + SYSTEM_SHARE, THIS_ONE.ram)

    def test_it_is_a_whole_number_of_gibibytes(self):
        for ram in SIZES:
            with self.subTest(ram=ram):
                given = system_memory(Fitted(), machine(ram), Mib(1000))

                self.assertEqual(given.cache % 1024, 0)

    def test_heavier_weights_never_leave_more(self):
        given = [system_memory(Fitted(), THIS_ONE, Mib(held)).cache
                 for held in range(0, 40000, 517)]

        self.assertEqual(given, sorted(given, reverse=True))

    def test_a_machine_with_nothing_left_holds_nothing(self):
        """Not a negative number, and not a refusal: the models still run."""
        given = system_memory(Fitted(), machine(16384), Mib(12000))

        self.assertEqual(Mib(0), given.cache)

    def test_more_than_half_of_a_large_machine_is_left_for_prefixes(self):
        """The reason a share of one half stopped being the rule."""
        given = system_memory(Fitted(), THIS_ONE, Mib(12000))

        self.assertGreater(given.cache, THIS_ONE.ram // 2)


class TheAnswerSaysWhatItWasWorkedOutFrom(unittest.TestCase):
    def test_the_three_figures_belong_to_one_reading(self):
        given = system_memory(Fitted(), THIS_ONE, Mib(12159))

        self.assertEqual(SystemMemory(installed=THIS_ONE.ram, resident=Mib(12159),
                                      cache=given.cache),
                         given)


class OnlyTheFastCoresCount(unittest.TestCase):
    """The pool is synchronised by a barrier; a slow thread holds up every other."""

    def test_slow_cores_add_nothing(self):
        self.assertEqual(threads(DESKTOP), threads(LAPTOP))

    def test_the_count_is_of_logical_processors_not_of_cores(self):
        self.assertEqual(threads(DESKTOP), 16)
        self.assertEqual(threads([Core(1, 1)] * 8 + [Core(0, 1)] * 8), 8)

    def test_one_class_of_core_means_all_of_them(self):
        self.assertEqual(threads(NO_HYBRID), 12)

    def test_adding_slow_cores_never_changes_the_answer(self):
        for slow in range(0, 33):
            with self.subTest(slow=slow):
                cores = list(DESKTOP) + [Core(0, 1)] * slow

                self.assertEqual(threads(cores), 16)

    def test_which_class_is_fast_is_read_from_the_cores_not_assumed(self):
        """Class numbers are whatever the machine reports; the highest is the fastest."""
        odd = [Core(7, 2)] * 4 + [Core(3, 1)] * 4

        self.assertEqual(threads(odd), 8)


class TheCardIsReadOffTheLineNvidiaSmiPrints(unittest.TestCase):
    """The real line, as this machine's driver prints it with noheader and nounits."""

    LINE = "16303, 13501, NVIDIA GeForce RTX 5070 Ti\n"

    def test_the_three_fields_are_told_apart(self):
        given = parse_occupancy(self.LINE)

        self.assertEqual("NVIDIA GeForce RTX 5070 Ti", given.first.card.name)
        self.assertEqual(Mib(16303), given.first.card.total)
        self.assertEqual(Mib(13501), given.first.free)

    def test_a_name_with_a_comma_in_it_does_not_become_a_number(self):
        """Not the whole line split on commas: a card is named, not counted."""
        with self.assertRaises(UnreadableDevice):
            parse_occupancy("16303, 13501, Some, Card\n")

    def test_every_card_is_read_in_the_order_printed(self):
        two = self.LINE + "8192, 8000, NVIDIA GeForce GTX 1080\n"

        self.assertEqual([Mib(16303), Mib(8192)],
                         [one.card.total for one in parse_occupancy(two)])

    def test_a_driver_that_says_nothing_useful_is_refused(self):
        for text in ("", "\n", "no devices were found\n",
                     "16303, 13501\n",
                     "N/A, N/A, NVIDIA GeForce RTX 5070 Ti\n"):
            with self.subTest(text=text):
                with self.assertRaises(UnreadableDevice):
                    parse_occupancy(text)

    def test_the_refusal_quotes_what_was_read(self):
        """Whatever went wrong, the line it went wrong on is in the message."""
        with self.assertRaises(UnreadableDevice) as refusal:
            parse_occupancy("driver not loaded")

        self.assertIn("driver not loaded", str(refusal.exception))


class WeightsOffTheCardGetWhatTheSystemAndTheCacheLeave(unittest.TestCase):
    """A mixture is placed by moving experts into system memory, and the room for them
    is what the machine has once the system and a declared cache have had theirs."""

    def test_a_fitted_cache_asks_for_none_of_it(self):
        """It is what is left after the weights, so it does not bound them in turn."""
        self.assertEqual(THIS_ONE.ram - SYSTEM_SHARE, off_card(Fitted(), THIS_ONE))

    def test_a_size_written_down_comes_out_of_it(self):
        self.assertEqual(THIS_ONE.ram - SYSTEM_SHARE - Mib(32768),
                         off_card(Fixed(Mib(32768)), THIS_ONE))

    def test_a_percentage_comes_out_of_it_as_the_cache_reads_it(self):
        asked = system_memory(Share(50), THIS_ONE, Mib(0)).cache

        self.assertEqual(THIS_ONE.ram - SYSTEM_SHARE - asked,
                         off_card(Share(50), THIS_ONE))

    def test_a_cache_larger_than_the_machine_leaves_no_room_rather_than_less(self):
        for ram in SIZES:
            with self.subTest(ram=ram):
                self.assertEqual(Mib(0), off_card(Fixed(Mib(ram)), machine(ram)))

    def test_the_room_the_cache_and_the_system_are_the_whole_machine(self):
        """Nothing of a machine that has room for all three is left unaccounted for."""
        for ram in SIZES:
            asked = Mib(ram // 4)
            if ram < asked + SYSTEM_SHARE:
                continue
            with self.subTest(ram=ram):
                room = off_card(Fixed(asked), machine(ram))

                self.assertEqual(Mib(ram), room + asked + SYSTEM_SHARE)

class AMachineHasCores(unittest.TestCase):
    def test_one_without_them_cannot_be_built(self):
        with self.assertRaises(ValueError):
            Machine(cards=installed(Card("test", Mib(16303))), ram=Mib(65407), cores=())


if __name__ == "__main__":
    unittest.main()
