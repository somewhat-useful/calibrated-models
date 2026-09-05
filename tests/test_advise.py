"""Invariants of what `vram` proposes closing.

The proposal is acted on by closing things, so being wrong costs a person their work.
It is taken from the big end and a browser is taken first, and both of those are about
what is lost rather than about megabytes -- so most of these check the order rather than
the arithmetic.

Nothing here checks that a hopeless case is reported as hopeless, because there is no
such answer to give. What a window is credited with is a floor on what closing it hands
back and never a ceiling, so where the floor does not reach, everything is proposed.
"""

import unittest

from cm.advise import (Fits, Holder, OnScreen, OutOfSight, Pid, Room, Short,
                       Standing, order, standing)
from cm.advise import to_close
from cm.units import Mib

TELEGRAM = Holder("Telegram.exe", Pid(7720), Mib(96), OnScreen())
CHROME = Holder("chrome.exe", Pid(1912), Mib(184), OnScreen())
CODE = Holder("Code.exe", Pid(9044), Mib(312), OnScreen())
OBS = Holder("obs64.exe", Pid(13388), Mib(740), OnScreen())

DESKTOP = (TELEGRAM, CHROME, CODE, OBS)

# Everything this desktop holds, which is as far as closing it can be said to go.
ALL_OF_IT = Mib(1332)


def room(free, wanted=1024) -> Room:
    return Room(free=Mib(free), wanted=Mib(wanted))


def held(holders) -> Mib:
    return Mib(sum(one.held for one in holders))


class WhatIsAlreadyFreeIsLeftAlone(unittest.TestCase):
    def test_room_to_spare_proposes_nothing(self):
        self.assertEqual((), to_close(room(free=2000), DESKTOP))

    def test_exactly_what_is_wanted_is_enough(self):
        """The figure is what should be free, not what should be exceeded."""
        self.assertEqual((), to_close(room(free=1024), DESKTOP))

    def test_a_card_nothing_is_holding_proposes_nothing(self):
        self.assertEqual((), to_close(room(free=1024), ()))


class TheBrowserGoesFirst(unittest.TestCase):
    """It comes back with its tabs where they were, which nothing else on a desktop
    does. That is worth more than the megabytes another window would have freed."""

    def test_it_is_proposed_before_a_window_holding_four_times_as_much(self):
        given = to_close(room(free=763), DESKTOP)

        self.assertEqual(CHROME, given[0])

    def test_it_is_the_whole_proposal_where_it_covers_the_shortfall(self):
        self.assertEqual((CHROME,), to_close(room(free=924), DESKTOP))

    def test_a_desktop_without_one_starts_at_the_largest(self):
        given = to_close(room(free=763), (TELEGRAM, CODE, OBS))

        self.assertEqual(OBS, given[0])

    def test_two_browsers_come_before_everything_and_the_larger_of_them_first(self):
        edge = Holder("msedge.exe", Pid(2000), Mib(90), OnScreen())

        self.assertEqual((CHROME, edge, OBS, CODE, TELEGRAM),
                         order(DESKTOP + (edge,)))


class TheRestIsTakenFromTheBigEnd(unittest.TestCase):
    def test_the_order_is_the_browser_then_falling_size(self):
        self.assertEqual((CHROME, OBS, CODE, TELEGRAM), order(DESKTOP))

    def test_what_is_proposed_follows_that_order(self):
        for free in range(0, 1025, 37):
            with self.subTest(free=free):
                given = to_close(room(free=free), DESKTOP)

                self.assertEqual(order(DESKTOP)[:len(given)], given)

    def test_a_window_holding_nothing_is_never_proposed(self):
        idle = Holder("Notepad.exe", Pid(400), Mib(0), OnScreen())

        self.assertNotIn(idle, to_close(room(free=0), DESKTOP + (idle,)))

    def test_a_window_holding_nothing_is_not_in_the_order_at_all(self):
        idle = Holder("Notepad.exe", Pid(400), Mib(0), OnScreen())

        self.assertNotIn(idle, order(DESKTOP + (idle,)))


class NothingIsProposedThatIsNotNeeded(unittest.TestCase):
    def test_the_last_one_taken_is_the_one_that_covers_it(self):
        """Without it the proposal would leave the model short, so nothing above it in
        the list is there for nothing."""
        for free in range(0, 1025, 37):
            given = to_close(room(free=free), DESKTOP)
            if not given:
                continue
            with self.subTest(free=free):
                self.assertLess(held(given[:-1]), room(free=free).wanted - free)

    def test_a_shortfall_one_window_covers_is_one_window(self):
        self.assertEqual(1, len(to_close(room(free=1000), DESKTOP)))

    def test_everything_proposed_was_on_the_list(self):
        for one in to_close(room(free=0), DESKTOP):
            self.assertIn(one, DESKTOP)


class WhereTheFiguresDoNotReachEverythingIsProposed(unittest.TestCase):
    def test_the_whole_desktop_is_the_proposal(self):
        given = to_close(room(free=0, wanted=4000), DESKTOP)

        self.assertEqual(order(DESKTOP), given)

    def test_the_last_megabyte_changes_nothing_about_the_answer(self):
        """1332 is everything this desktop holds. One megabyte more is not a different
        kind of answer, only a shortfall these figures do not reach."""
        for wanted in (ALL_OF_IT, Mib(ALL_OF_IT + 1)):
            with self.subTest(wanted=wanted):
                self.assertEqual(order(DESKTOP),
                                 to_close(room(free=0, wanted=wanted), DESKTOP))

    def test_nothing_open_proposes_nothing(self):
        self.assertEqual((), to_close(room(free=0), ()))

    def test_windows_holding_nothing_are_the_same_as_no_windows(self):
        idle = (Holder("Notepad.exe", Pid(400), Mib(0), OnScreen()),
                Holder("Calculator.exe", Pid(401), Mib(0), OnScreen()))

        self.assertEqual((), to_close(room(free=0), idle))


class WhetherItFitsIsAskedOnItsOwn(unittest.TestCase):
    """A screen listing many models asks this of every one of them, and asks what to
    close about only the one a person picked."""

    def test_it_says_what_is_over_when_it_fits(self):
        given = standing(room(free=2000))

        self.assertIsInstance(given, Fits)
        self.assertEqual(Mib(976), given.spare)

    def test_it_says_what_is_missing_when_it_does_not(self):
        given = standing(room(free=763))

        self.assertIsInstance(given, Short)
        self.assertEqual(Mib(261), given.by)

    def test_exactly_enough_fits(self):
        self.assertIsInstance(standing(room(free=1024)), Fits)

    def test_every_answer_is_one_of_the_two(self):
        for free in range(0, 2000, 13):
            with self.subTest(free=free):
                self.assertIsInstance(standing(room(free=free)), Standing)

    def test_it_never_disagrees_with_what_is_proposed(self):
        """Two ways of asking one question, and a screen puts both on it at once."""
        for free in range(0, 2000, 13):
            with self.subTest(free=free):
                self.assertEqual(isinstance(standing(room(free=free)), Fits),
                                 to_close(room(free=free), DESKTOP) == ())


class HowAProgramClosesDoesNotChangeWhatIsProposed(unittest.TestCase):
    """Whether a program is asked or ended is settled elsewhere, by what it has on
    screen. Here it counts for what it holds, like anything else."""

    def test_a_program_with_nothing_on_screen_is_proposed(self):
        """It cannot be asked, but it can be ended, and that frees the card."""
        tray = Holder("SamsungMagician.exe", Pid(21904), Mib(900), OutOfSight())

        self.assertEqual((tray,), order((tray,)))

    def test_the_order_is_by_what_is_held_either_way(self):
        tray = Holder("SamsungMagician.exe", Pid(21904), Mib(900), OutOfSight())

        self.assertEqual((CHROME, tray, OBS, CODE),
                         order((OBS, tray, CHROME, CODE)))

    def test_a_shortfall_is_covered_out_of_both_kinds(self):
        tray = Holder("SamsungMagician.exe", Pid(21904), Mib(400), OutOfSight())
        room = Room(free=Mib(1000), wanted=Mib(1500))

        self.assertEqual((CHROME, tray), to_close(room, (tray, CHROME)))


if __name__ == "__main__":
    unittest.main()
