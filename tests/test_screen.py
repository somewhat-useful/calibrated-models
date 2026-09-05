"""Invariants of the `vram` screen.

A person acts on what this says, so every number on it has to be the one that was
decided and every window that could be closed has to be on it. The drawing is checked
as text: what a terminal does with it afterwards is not what can be wrong here.

The screen carries two lists at once, and most of what can go wrong is between them --
a profile said to fit while windows below it are marked for closing, or a count that
answers about a profile other than the one picked. That is what the middle classes here
are about.
"""

import unittest

from cm.advise import Holder, OnScreen, OutOfSight, Pid, Unoffered
from cm.catalog import Loadable
from cm.machine import Card, Occupancy
from cm.screen import (DOWN, KEYS, MARK, NEXT, PREV, UP, Rows, lines, moved,
                       opens_on, picked, toggled)
from cm.units import Mib

CARD = Card("NVIDIA GeForce RTX 5070 Ti", Mib(16303))

GEMMA = Loadable("gemma4-12b", Mib(10285))
ORNITH = Loadable("ornith-1.0-35b", Mib(15335))
QWEN = Loadable("qwen3.8-45k-q8", Mib(15276))

OFFERED = (GEMMA, ORNITH, QWEN)

OBS = Holder("obs64.exe", Pid(13388), Mib(740), OnScreen())
CODE = Holder("Code.exe", Pid(9044), Mib(312), OnScreen())
CHROME = Holder("chrome.exe", Pid(1912), Mib(184), OnScreen())
TELEGRAM = Holder("Telegram.exe", Pid(7720), Mib(96), OnScreen())

DESKTOP = (OBS, CODE, CHROME, TELEGRAM)

# Holding the card with no window up: ended rather than asked.
TRAY = Holder("SamsungMagician.exe", Pid(21904), Mib(252), OutOfSight())

# Longer than any console window.
CROWD = tuple(Holder(f"program{n}.exe", Pid(1000 + n), Mib(500 - n), OnScreen())
              for n in range(30))


def screen(free, router=0, offered=OFFERED, chosen=ORNITH, holders=DESKTOP,
           cursor=Pid(9044), marked=frozenset(), unoffered=Mib(0),
           this_window=Mib(0), room=Rows(40)):
    return lines(Occupancy(card=CARD, free=Mib(free)), Mib(free + router),
                 offered, chosen, holders, cursor, marked,
                 Unoffered(held=unoffered, this_window=this_window), room)


def text(free, **rest) -> str:
    return "\n".join(one.text for one in screen(free, **rest))


class TheHeaderSaysWhatTheCardIsDoing(unittest.TestCase):
    def test_the_three_numbers_are_the_total_the_used_and_the_free(self):
        head = screen(free=13401)[0].text

        self.assertIn("16303", head)
        self.assertIn("2902", head)
        self.assertIn("13401", head)

    def test_what_is_in_use_is_the_rest_of_the_card_and_not_a_reading(self):
        self.assertIn("2802", screen(free=13501)[0].text)

    def test_the_card_says_which_one_it_is(self):
        self.assertIn(CARD.name, screen(free=13401)[0].text)


class WhatTheRouterHoldsIsAlreadyTheNextModelsMemory(unittest.TestCase):
    """It keeps one model resident and unloads it before it loads another. Counting
    that as taken would report a model that is loaded right now as one that does not
    fit."""

    def test_a_profile_is_measured_against_the_card_plus_what_the_router_holds(self):
        self.assertIn("loads now, 43 MiB to spare", text(free=15200, router=178))

    def test_the_same_card_without_the_router_leaves_it_short(self):
        self.assertIn("short by 135 MiB", text(free=15200))

    def test_what_the_router_holds_is_said_and_so_is_the_sum(self):
        said = screen(free=13401, router=178)[1].text

        self.assertIn("178", said)
        self.assertIn("13579", said)

    def test_a_router_holding_nothing_says_nothing(self):
        self.assertNotIn("router", text(free=13401))


class EveryProfileIsOnTheScreenWithWhatItCosts(unittest.TestCase):
    def test_each_one_gives_its_name_and_what_it_holds(self):
        drawn = text(free=13401)

        for one in OFFERED:
            with self.subTest(profile=one.name):
                self.assertIn(one.name, drawn)
                self.assertIn(str(one.needs), drawn)

    def test_they_are_drawn_in_the_order_they_arrive(self):
        drawn = text(free=13401)
        found = [drawn.index(one.name) for one in OFFERED]

        self.assertEqual(sorted(found), found)

    def test_one_that_fits_says_what_is_left_over(self):
        self.assertIn("loads now, 3116 MiB to spare", text(free=13401))

    def test_one_that_does_not_says_how_much_is_missing(self):
        drawn = text(free=13401)

        self.assertIn("short by 1934 MiB", drawn)
        self.assertIn("short by 1875 MiB", drawn)

    def test_what_is_missing_is_worked_out_again_on_every_reading(self):
        """The point of the screen is to watch these fall as windows close."""
        self.assertIn("short by 335 MiB", text(free=15000))
        self.assertIn("short by 235 MiB", text(free=15100))


class ThePickedProfileIsWhatTheCountingIsAbout(unittest.TestCase):
    def test_exactly_one_profile_is_marked_as_picked(self):
        marked = [one.text for one in screen(free=15000) if one.text.startswith(">")]

        self.assertEqual(1, len(marked))
        self.assertIn(ORNITH.name, marked[0])

    def test_what_is_still_missing_is_the_picked_profiles_shortfall(self):
        """Short by 335 for ornith and by 276 for qwen, with the same windows open."""
        self.assertIn("335 needed", text(free=15000, chosen=ORNITH,
                                         marked=frozenset({CHROME.pid})))
        self.assertIn("276 needed", text(free=15000, chosen=QWEN,
                                         marked=frozenset({CHROME.pid})))

    def test_picking_one_that_already_fits_leaves_nothing_to_say(self):
        drawn = screen(free=15000, chosen=GEMMA)

        self.assertEqual(KEYS, drawn[-1].text)
        self.assertEqual("", drawn[-2].text)

    def test_the_other_profiles_are_still_listed_with_their_standing(self):
        drawn = text(free=15000, chosen=GEMMA)

        self.assertIn("short by 335 MiB", drawn)
        self.assertIn("short by 276 MiB", drawn)


class EveryWindowIsOnTheScreen(unittest.TestCase):
    def test_each_one_gives_its_name_its_process_and_what_it_holds(self):
        drawn = text(free=13401)

        for holder in DESKTOP:
            with self.subTest(holder=holder.name):
                self.assertIn(holder.name, drawn)
                self.assertIn(str(holder.pid), drawn)
                self.assertIn(str(holder.held), drawn)

    def test_they_are_drawn_in_the_order_they_arrive(self):
        drawn = text(free=13401)
        found = [drawn.index(holder.name) for holder in DESKTOP]

        self.assertEqual(sorted(found), found)

    def test_a_card_with_nothing_on_it_still_draws(self):
        drawn = screen(free=13401, holders=())

        self.assertEqual(KEYS, drawn[-1].text)
        self.assertIn("nothing here that can be given up", drawn[-2].text)


class EveryWindowCarriesABoxSayingWhetherItGoes(unittest.TestCase):
    def test_a_marked_window_is_ticked_and_the_rest_are_not(self):
        drawn = [one.text for one in screen(free=15000, marked=frozenset({OBS.pid}))]

        self.assertEqual(1, len([one for one in drawn if "[x]" in one]))
        self.assertIn("obs64.exe", [one for one in drawn if "[x]" in one][0])

    def test_every_window_has_a_box_either_way(self):
        drawn = [one.text for one in screen(free=15000, marked=frozenset({OBS.pid}))]
        boxed = [one for one in drawn if "[x]" in one or "[ ]" in one]

        self.assertEqual(len(DESKTOP), len(boxed))

    def test_a_mark_on_a_window_that_is_not_there_ticks_nothing(self):
        drawn = text(free=15000, marked=frozenset({Pid(999)}))

        self.assertNotIn("[x]", drawn)


class WhatTheMarksWouldComeToIsSaidUnderThem(unittest.TestCase):
    def test_marking_enough_says_so(self):
        said = screen(free=15000, marked=frozenset({OBS.pid}))[-2].text

        self.assertIn("marked 1 program, 740 MiB", said)
        self.assertIn("enough", said)

    def test_marking_too_little_says_how_far_it_gets(self):
        said = screen(free=15000, marked=frozenset({CHROME.pid}))[-2].text

        self.assertIn("marked 1 program, 184 MiB of the 335 needed", said)

    def test_and_says_why_that_is_not_the_whole_story(self):
        """Marks that do not add up to the shortfall look like a bug unless the screen
        says why they are not one."""
        self.assertIn("compositor",
                      screen(free=15000, marked=frozenset({CHROME.pid}))[-2].text)

    def test_several_marks_are_added_up(self):
        said = screen(free=15000,
                      marked=frozenset({CHROME.pid, CODE.pid}))[-2].text

        self.assertIn("marked 2 programs, 496 MiB", said)

    def test_marking_nothing_still_says_what_is_missing(self):
        self.assertIn("nothing marked, and 335 MiB still to free",
                      screen(free=15000)[-2].text)

    def test_a_profile_that_fits_says_nothing_about_the_marks(self):
        """The boxes stay where they are; there is just nothing to count them against."""
        said = [one.text for one in screen(free=15400, marked=frozenset({OBS.pid}))]

        self.assertEqual([], [one for one in said if "marked 1 program" in one])


class TheCursorIsOnOneLineAtMost(unittest.TestCase):
    def test_it_sits_on_the_window_it_names(self):
        under = [one.text for one in screen(free=13401, cursor=OBS.pid) if one.cursor]

        self.assertEqual(1, len(under))
        self.assertIn("obs64.exe", under[0])

    def test_a_cursor_on_nothing_leaves_every_line_alone(self):
        drawn = screen(free=13401, cursor=Pid(1))

        self.assertEqual([], [one for one in drawn if one.cursor])

    def test_no_profile_is_ever_under_it(self):
        """It says what a mark would land on, and a profile is not something to close."""
        under = [one.text for one in screen(free=13401, cursor=OBS.pid) if one.cursor]

        for one in OFFERED:
            with self.subTest(profile=one.name):
                self.assertNotIn(one.name, under[0])

    def test_the_header_and_the_keys_are_never_under_it(self):
        drawn = screen(free=13401)

        self.assertFalse(drawn[0].cursor)
        self.assertFalse(drawn[-1].cursor)


class TheKeysAreAlwaysTheLastThingSaid(unittest.TestCase):
    def test_they_close_every_screen(self):
        for free in (13401, 15000, 15400):
            with self.subTest(free=free):
                self.assertEqual(KEYS, screen(free=free)[-1].text)

    def test_every_one_of_them_is_named(self):
        for key in ("tab", "shift-tab", "space", "enter", "r", "q"):
            with self.subTest(key=key):
                self.assertIn(f"[{key}]", KEYS)


class TheScreenOpensOnSomethingWorthDoing(unittest.TestCase):
    def test_it_opens_on_the_first_profile_that_does_not_fit(self):
        self.assertEqual(ORNITH, opens_on(Mib(15000), OFFERED))

    def test_a_profile_that_already_loads_is_passed_over(self):
        """gemma is first on the list and fits; the screen opens past it."""
        self.assertNotEqual(GEMMA, opens_on(Mib(15300), OFFERED))

    def test_where_everything_fits_it_opens_on_the_largest(self):
        """The first one to stop fitting as the desktop grows."""
        self.assertEqual(ORNITH, opens_on(Mib(16000), OFFERED))

    def test_a_preset_offering_nothing_is_not_a_screen(self):
        with self.assertRaises(ValueError):
            opens_on(Mib(16000), ())


class ThePickWalksTheListAndComesBackRound(unittest.TestCase):
    def test_it_moves_one_on(self):
        self.assertEqual(ORNITH, picked(GEMMA, NEXT, OFFERED))

    def test_it_comes_back_to_the_first(self):
        """Unlike the cursor: a ring of things to pick, not a column to walk."""
        self.assertEqual(GEMMA, picked(QWEN, NEXT, OFFERED))

    def test_it_moves_one_back(self):
        self.assertEqual(GEMMA, picked(ORNITH, PREV, OFFERED))

    def test_going_back_from_the_first_comes_round_to_the_last(self):
        self.assertEqual(QWEN, picked(GEMMA, PREV, OFFERED))

    def test_back_undoes_on_wherever_it_is_asked(self):
        for one in OFFERED:
            with self.subTest(profile=one.name):
                self.assertEqual(one, picked(picked(one, NEXT, OFFERED), PREV, OFFERED))

    def test_an_empty_list_is_not_walked_backwards_either(self):
        self.assertEqual(QWEN, picked(QWEN, PREV, ()))

    def test_any_other_key_leaves_it_where_it_was(self):
        for key in (UP, DOWN, MARK, "enter", "r", ""):
            with self.subTest(key=key):
                self.assertEqual(QWEN, picked(QWEN, key, OFFERED))

    def test_a_pick_no_longer_offered_lands_on_the_first(self):
        """A preset rewritten under a running screen is not a reason to fall over."""
        gone = Loadable("qwen3.8-88k-q4", Mib(15279))

        self.assertEqual(GEMMA, picked(gone, NEXT, OFFERED))

    def test_an_empty_list_leaves_it_alone(self):
        self.assertEqual(QWEN, picked(QWEN, NEXT, ()))


class MarkingTakesOneWindowInOrOut(unittest.TestCase):
    def test_it_takes_in_what_was_not_marked(self):
        self.assertEqual(frozenset({OBS.pid}),
                         toggled(frozenset(), MARK, OBS))

    def test_it_takes_out_what_was(self):
        self.assertEqual(frozenset({CODE.pid}),
                         toggled(frozenset({CODE.pid, OBS.pid}), MARK, OBS))

    def test_it_leaves_the_others_where_they_were(self):
        given = toggled(frozenset({CODE.pid}), MARK, OBS)

        self.assertIn(CODE.pid, given)

    def test_any_other_key_changes_nothing(self):
        for key in (UP, DOWN, NEXT, PREV, "enter", "r", ""):
            with self.subTest(key=key):
                self.assertEqual(frozenset({CODE.pid}),
                                 toggled(frozenset({CODE.pid}), key, OBS))


class TheCursorStopsAtTheEnds(unittest.TestCase):
    def test_it_walks_the_list(self):
        self.assertEqual(2, moved(1, DOWN, 4))
        self.assertEqual(0, moved(1, UP, 4))

    def test_it_does_not_come_back_round(self):
        self.assertEqual(0, moved(0, UP, 4))
        self.assertEqual(3, moved(3, DOWN, 4))

    def test_a_key_that_is_not_a_direction_leaves_it_where_it_was(self):
        for key in ("enter", "r", NEXT, PREV, MARK, ""):
            with self.subTest(key=key):
                self.assertEqual(2, moved(2, key, 4))

    def test_an_empty_list_has_nowhere_to_stand(self):
        for key in (UP, DOWN, "enter"):
            with self.subTest(key=key):
                self.assertEqual(0, moved(0, key, 0))

    def test_a_cursor_left_past_the_end_comes_back_onto_the_list(self):
        """Closing a window shortens the list under a cursor that was below it."""
        self.assertEqual(1, moved(6, DOWN, 2))


def drawn_rows(drawn, among=CROWD):
    """The holder rows of a frame, by the names of the programs on it."""
    names = {one.name for one in among}
    return [one.text for one in drawn
            if any(name in one.text for name in names)]


class ARowIsABoxANameAndANumber(unittest.TestCase):
    """And nothing else. How a program is closed follows from what it has on screen and
    is not a choice anybody makes here, so a row saying it is a row read for nothing."""

    def test_a_program_with_a_window_up_is_a_box_a_name_and_a_number(self):
        self.assertIn("[ ] obs64.exe", text(free=13401))

    def test_a_program_with_nothing_on_screen_reads_the_same(self):
        self.assertIn("[ ] SamsungMagician.exe", text(free=13401, holders=(TRAY,)))

    def test_a_row_ends_at_its_number(self):
        rows = drawn_rows(screen(free=13401, holders=(TRAY,)), among=(TRAY,))
        self.assertEqual(1, len(rows))
        self.assertTrue(rows[0].rstrip().endswith("MiB"), rows[0])

    def test_a_row_that_is_ended_marks_like_any_other(self):
        self.assertEqual(frozenset({TRAY.pid}), toggled(frozenset(), MARK, TRAY))

    def test_a_screen_with_nothing_to_give_up_says_so(self):
        self.assertIn("nothing here that can be given up",
                      text(free=13401, holders=()))


class WhatIsNotOnOfferIsOneLine(unittest.TestCase):
    """Rows nobody can act on are rows to scroll past, but the figure cannot be left out
    either: without it the header says a gigabyte is in use and the list accounts for a
    tenth of it."""

    def test_it_is_said_under_the_list(self):
        self.assertIn("the other 320 MiB is Windows, the compositor and this window",
                      text(free=13401, unoffered=Mib(320)))

    def test_it_is_not_said_where_nothing_is_left_over(self):
        self.assertNotIn("the other", text(free=13401, unoffered=Mib(0)))

    def test_it_is_not_a_row_and_takes_no_cursor(self):
        drawn = screen(free=13401, unoffered=Mib(320))
        said = [one for one in drawn if "the other" in one.text]

        self.assertEqual([False], [one.cursor for one in said])

    def test_it_costs_one_line_of_the_room_and_no_more(self):
        with_it = len(screen(free=13401, unoffered=Mib(320), room=Rows(40)))
        without = len(screen(free=13401, unoffered=Mib(0), room=Rows(40)))

        self.assertEqual(without + 1, with_it)


class WhatThisWindowItselfHoldsIsSaidToo(unittest.TestCase):
    """It is inside the figure above it and comes back all the same: the model is
    loaded from here, and then this window is closed like any other."""

    def test_it_says_what_comes_back_when_this_window_closes(self):
        self.assertIn("118 MiB of that is this window, and comes back when it is closed",
                      text(free=13401, unoffered=Mib(320), this_window=Mib(118)))

    def test_it_is_not_said_where_this_window_holds_nothing(self):
        self.assertNotIn("this window, and comes back",
                         text(free=13401, unoffered=Mib(320), this_window=Mib(0)))

    def test_it_is_not_said_where_there_is_nothing_left_over_at_all(self):
        """It is part of that figure, and a part of nothing is nothing to say."""
        self.assertNotIn("comes back",
                         text(free=13401, unoffered=Mib(0), this_window=Mib(118)))

    def test_it_is_not_a_row_and_cannot_be_marked(self):
        """Marking it would end the program in the middle of being asked."""
        drawn = screen(free=13401, unoffered=Mib(320), this_window=Mib(118))
        said = [one for one in drawn if "comes back" in one.text]

        self.assertEqual([False], [one.cursor for one in said])
        self.assertNotIn("[ ]", said[0].text)

    def test_it_costs_one_line_of_the_room_and_no_more(self):
        with_it = len(screen(free=13401, unoffered=Mib(320), this_window=Mib(118),
                             room=Rows(40)))
        without = len(screen(free=13401, unoffered=Mib(320), room=Rows(40)))

        self.assertEqual(without + 1, with_it)


class TheFrameFitsTheConsoleWindow(unittest.TestCase):
    """A frame taller than the window scrolls it, and the next frame drawn from the top
    lands in the middle of the last one. What that looks like is a screen that has
    stopped answering."""

    def test_it_never_draws_more_lines_than_it_was_given(self):
        for room in range(1, 45):
            with self.subTest(room=room):
                self.assertLessEqual(
                    len(screen(free=13401, holders=CROWD, cursor=CROWD[0].pid,
                               room=Rows(room))),
                    room)

    def test_with_room_to_spare_every_program_is_drawn(self):
        given = screen(free=13401, holders=CROWD, cursor=CROWD[0].pid, room=Rows(60))

        self.assertEqual(len(CROWD), len(drawn_rows(given)))
        self.assertNotIn("more", given[-1].text)

    def test_what_does_not_fit_is_counted_and_not_dropped_silently(self):
        given = screen(free=13401, holders=CROWD, cursor=CROWD[0].pid, room=Rows(24))
        shown = len(drawn_rows(given))
        said = [one.text for one in given if "and" in one.text and "more" in one.text]

        self.assertEqual([f"      ... and {len(CROWD) - shown} more holding less"], said)

    def test_the_cursor_is_on_the_screen_wherever_it_is_in_the_list(self):
        for at in (0, 7, 15, len(CROWD) - 1):
            with self.subTest(at=at):
                given = screen(free=13401, holders=CROWD, cursor=CROWD[at].pid,
                               room=Rows(24))

                self.assertEqual([True], [one.cursor for one in given if one.cursor])
                self.assertIn(CROWD[at].name,
                              "".join(one.text for one in given if one.cursor))

    def test_the_top_of_the_list_is_kept_where_it_fits(self):
        """The answer to what to give up is at the top, so it is what stays."""
        given = screen(free=13401, holders=CROWD, cursor=CROWD[0].pid, room=Rows(24))

        self.assertIn(CROWD[0].name, drawn_rows(given)[0])

    def test_the_card_and_the_keys_are_never_the_lines_dropped(self):
        given = screen(free=13401, holders=CROWD, cursor=CROWD[0].pid, room=Rows(20))

        self.assertIn(CARD.name, given[0].text)
        self.assertEqual(KEYS, given[-1].text)


if __name__ == "__main__":
    unittest.main()
