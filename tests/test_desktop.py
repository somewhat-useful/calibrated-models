"""Invariants of gathering video memory under the programs holding it.

What holds the memory is rarely what a person would close: chrome draws in children of
its own, and the compositor is credited with surfaces belonging to every window on
screen. So the question is not who allocated it but who would give it back -- and the
answer is a program, whether or not it has a window anywhere on the screen.
"""

import unittest

from cm.advise import Holder, Pid
from cm.advise import OnScreen, OutOfSight
from cm.desktop import (DESKTOP_SHARE, Running, closes, desktops, family,
                        held_by_this_window, holders, not_offered)
from cm.machine import Card, Occupancy
from cm.units import Mib

CARD = Card("NVIDIA GeForce RTX 5070 Ti", Mib(16303))


def process(pid, name, parent=0, held=0, windowed=False, system=False,
            microsoft=False, publisher="") -> Running:
    return Running(pid=Pid(pid), name=name, parent=Pid(parent),
                   held=Mib(held), windowed=windowed, system=system,
                   microsoft=system or microsoft, publisher=publisher)


# The terminal this program is drawn in, and the program itself. Both are in a snapshot
# like anything else; neither may be offered, because accepting would end the program.
TERMINAL = process(40564, "WindowsTerminal.exe", parent=13864, held=85, windowed=True)
US = process(50120, "python.exe", parent=40564)

# Somewhere this program is not running from, for asking what the list says without it.
ELSEWHERE = Pid(1)


# What this machine actually reported, cut down to the processes that matter.
DESKTOP = (
    process(2224, "dwm.exe", parent=1776, held=3505),
    process(1776, "winlogon.exe"),
    # One window open, drawing in children of its own.
    process(35772, "chrome.exe", parent=13864, windowed=True),
    process(30780, "chrome.exe", parent=35772, held=197),
    process(33852, "chrome.exe", parent=35772, held=11),
    # Nothing on screen at all: minimised to the notification area, still holding.
    process(8108, "ChatGPT.exe", parent=13864),
    process(17040, "ChatGPT.exe", parent=8108, held=234),
    process(34036, "ChatGPT.exe", parent=8108, held=28),
    # The shell, which started most of the above, and one open folder window.
    process(13864, "explorer.exe", parent=13792, held=170),
    process(26284, "explorer.exe", parent=1996, held=59, windowed=True),
    process(1996, "svchost.exe", parent=1816),
    # The router, holding the model it has loaded.
    process(16200, "llama-server.exe", parent=3712),
    process(23676, "llama-server.exe", parent=16200, held=178),
    TERMINAL,
    US,
)


def credited(running, ours=US.pid):
    return {one.name: one.held for one in holders(running, ours)}


def _closing(running, pid, ours=US.pid):
    """How the row for this program says it would be closed, or None where there is no
    row: what Windows holds itself is a figure under the list, not a row on it."""
    found = [one.closing for one in holders(running, ours) if one.pid == pid]
    return found[0] if found else None


def under(running, pid, ours=US.pid):
    return [one.held for one in holders(running, ours) if one.pid == Pid(pid)]


class MemoryIsCreditedToTheProgramThatWouldGiveItBack(unittest.TestCase):
    def test_a_program_of_one_process_keeps_what_it_holds(self):
        alone = (process(26284, "explorer.exe", parent=1996, held=56),)

        self.assertEqual({"explorer.exe": Mib(56)}, credited(alone))

    def test_what_a_child_holds_is_for_the_program_to_give_back(self):
        """Chrome draws in children of its own; a person closes chrome."""
        self.assertEqual([Mib(197 + 11)], under(DESKTOP, 35772))

    def test_several_children_are_added_up_under_one_program(self):
        self.assertEqual([Mib(234 + 28)], under(DESKTOP, 8108))

    def test_the_topmost_process_of_the_same_program_is_the_one_credited(self):
        chain = (process(1, "app.exe", windowed=True),
                 process(2, "app.exe", parent=1),
                 process(3, "app.exe", parent=2, held=100))

        self.assertEqual([Mib(100)], under(chain, 1))

    def test_a_helper_of_the_same_publisher_is_part_of_the_program(self):
        """A program can draw its interface in a helper of another name. Ending that on
        its own is work the program undoes: it starts another."""
        helped = (process(24440, "app.exe", parent=16352, windowed=True,
                          publisher="One Publisher"),
                  process(22564, "interface.exe", parent=24440, held=46,
                          publisher="One Publisher"),
                  process(7332, "interface.exe", parent=22564, held=12,
                          publisher="One Publisher"))

        self.assertEqual({"app.exe": Mib(46 + 12)}, credited(helped))

    def test_a_program_started_by_another_publishers_is_its_own(self):
        """An editor that starts a browser has not become the browser."""
        pair = (process(9044, "Code.exe", held=20, publisher="Microsoft Corporation"),
                process(1912, "chrome.exe", parent=9044, held=184,
                        publisher="Google LLC"))

        self.assertEqual({"Code.exe": Mib(20), "chrome.exe": Mib(184)}, credited(pair))

    def test_a_file_that_names_no_publisher_claims_nobody_as_kin(self):
        """Two files saying nothing about who wrote them say nothing about belonging
        together either, whichever started which."""
        pair = (process(1, "launcher.exe", held=10),
                process(2, "worker.exe", parent=1, held=30))

        self.assertEqual({"launcher.exe": Mib(10), "worker.exe": Mib(30)},
                         credited(pair))

    def test_whoever_started_the_program_is_not_part_of_it(self):
        """Chrome and ChatGPT were both started from the shell, and neither is it."""
        self.assertEqual([Mib(170)], under(DESKTOP, 13864))

    def test_two_processes_of_one_name_started_apart_are_two_programs(self):
        """Both are explorer: the desktop itself, and a folder window."""
        self.assertEqual([Mib(170), Mib(59)],
                         [one.held for one in holders(DESKTOP, US.pid)
                          if one.name == "explorer.exe"])

    def test_a_process_whose_parent_is_gone_stands_for_itself(self):
        orphan = (process(30924, "claude.exe", parent=13660, held=119),)

        self.assertEqual({"claude.exe": Mib(119)}, credited(orphan))


class WhatHasNoWindowIsHoldingItAllTheSame(unittest.TestCase):
    """Most of what holds this card is minimised to the notification area. Leaving it
    out would leave out where the memory went."""

    def test_a_program_with_no_window_anywhere_is_on_the_list(self):
        self.assertEqual(Mib(234 + 28), credited(DESKTOP)["ChatGPT.exe"])

    def test_a_process_nobody_has_a_window_for_is_on_the_list(self):
        lone = (process(18028, "SearchHost.exe", parent=1996, held=13),
                process(1996, "svchost.exe", parent=1816))

        self.assertEqual({"SearchHost.exe": Mib(13)}, credited(lone))

    def test_holding_nothing_is_what_keeps_a_program_off_it(self):
        idle = (process(400, "notepad.exe", windowed=True),)

        self.assertEqual((), holders(idle, US.pid))


class EveryRowIsSomethingAPersonCanDo(unittest.TestCase):
    """Measured on this machine: Windows itself held twenty of the twenty-five programs
    on the card. Rows nobody can act on are rows to scroll past to reach the two that
    matter, so they are a figure under the list instead."""

    def test_a_program_with_a_window_up_is_asked(self):
        self.assertEqual(OnScreen(), _closing(DESKTOP, Pid(35772)))

    def test_a_program_with_nothing_on_screen_is_ended(self):
        """ChatGPT is in the notification area: there is no corner to click."""
        self.assertEqual(OutOfSight(), _closing(DESKTOP, Pid(8108)))

    def test_a_window_in_a_child_is_the_programs_window(self):
        two = (process(1, "app.exe"),
               process(2, "app.exe", parent=1, held=50, windowed=True))

        self.assertEqual(OnScreen(), _closing(two, Pid(1)))

    def test_windows_own_machinery_is_not_a_row_at_all(self):
        """A web view for the search box: Windows starts it again when it wants it."""
        webview = (process(18676, "msedgewebview2.exe", parent=18028, microsoft=True),
                   process(17924, "msedgewebview2.exe", parent=18676, held=81,
                           microsoft=True),
                   process(18028, "SearchHost.exe", system=True))

        self.assertEqual((), holders(webview, US.pid))

    def test_windows_own_with_a_window_up_is_a_row_like_any_other(self):
        """A folder window closes the way any window closes, and the shell behind it
        carries on. Closing means the window, never the program."""
        folder = (process(26284, "explorer.exe", parent=1996, held=59,
                          windowed=True, system=True),
                  process(1996, "svchost.exe", system=True))

        self.assertEqual(OnScreen(), _closing(folder, Pid(26284)))

    def test_microsofts_own_with_a_window_up_is_the_persons_to_close(self):
        """An editor and a browser are Microsoft's as often as anybody's."""
        editor = (process(9044, "Code.exe", held=312, windowed=True, microsoft=True),)

        self.assertEqual(OnScreen(), _closing(editor, Pid(9044)))

    def test_somebody_elses_with_nothing_on_screen_is_ended(self):
        tray = (process(21904, "SamsungMagician.exe", parent=13864, held=252),
                process(13864, "explorer.exe", system=True))

        self.assertEqual(OutOfSight(), _closing(tray, Pid(21904)))

    def test_what_windows_started_is_a_program_and_windows_is_not_part_of_it(self):
        """The shell starts most of what runs on a desktop. Counting what started a
        program as part of it would put the whole desktop under explorer, and leaving
        the program out because Windows started it would leave out everything."""
        started = (process(1, "explorer.exe", system=True),
                   process(2, "app.exe", parent=1, held=50))

        self.assertEqual(["app.exe"], [one.name for one in holders(started, US.pid)])


class WhatIsInUseAndNotOnTheListIsCountedBySubtraction(unittest.TestCase):
    """The per-process figures overlap: the compositor is credited with the surfaces of
    the windows drawing through it, and each surface is counted again under the program
    it belongs to. Measured on this machine: 1388 MiB across the processes against 1041
    in use on the card. Added up, the screen would disagree with its own header."""

    def test_it_is_what_is_in_use_beyond_what_the_list_offers(self):
        """Measured: 1041 in use, one row of 104, and 937 that nothing here can give."""
        card = Occupancy(card=CARD, free=Mib(15262))
        listed = (Holder("WindowsTerminal.exe", Pid(7032), Mib(104), OnScreen()),)

        self.assertEqual(Mib(937), not_offered(card, Mib(15262), listed))

    def test_what_the_router_hands_back_is_not_counted_into_it(self):
        """It is the next model's memory: the router unloads one before it loads the
        next, and `available` already says so."""
        card = Occupancy(card=CARD, free=Mib(16125))

        self.assertEqual(Mib(0), not_offered(card, Mib(16303), ()))

    def test_it_never_goes_below_nothing(self):
        """The card and the processes are read a moment apart and from different places,
        so the list can name memory the card has already been given back."""
        card = Occupancy(card=CARD, free=Mib(16000))
        listed = (Holder("app.exe", Pid(1), Mib(900), OnScreen()),)

        self.assertEqual(Mib(0), not_offered(card, Mib(16000), listed))

    def test_the_three_figures_come_to_what_the_card_says_is_in_use(self):
        """Which is the whole of it: a person checks the screen against its own header
        and the arithmetic is there to be done."""
        card = Occupancy(card=CARD, free=Mib(13404))
        available = Mib(13582)
        listed = holders(DESKTOP, US.pid)

        in_use = card.card.total - card.free
        offered = sum(one.held for one in listed)
        hands_back = available - card.free

        self.assertEqual(in_use,
                         offered + hands_back + not_offered(card, available, listed))


class WhatThisWindowHoldsIsCountedApart(unittest.TestCase):
    """It is not on the list -- accepting it would end the program in the middle of
    being asked -- but it does come back when the window is closed, so it is a figure
    of its own rather than nothing."""

    def test_it_is_what_the_window_this_runs_in_holds(self):
        self.assertEqual(Mib(85), held_by_this_window(DESKTOP, US.pid))

    def test_this_program_is_part_of_that_window(self):
        """The console holds the surfaces and this holds whatever it holds; both go
        together, because closing the window ends what is running in it."""
        here = (TERMINAL,
                process(50120, "python.exe", parent=TERMINAL.pid, held=7))

        self.assertEqual(Mib(85 + 7), held_by_this_window(here, US.pid))

    def test_this_program_alone_is_still_this_window(self):
        """Started with no window above it, what it holds is still not on offer."""
        alone = (process(50120, "python.exe", held=7),)

        self.assertEqual(Mib(7), held_by_this_window(alone, US.pid))
        self.assertEqual((), holders(alone, US.pid))

    def test_nobody_elses_window_is_counted_into_it(self):
        """Another terminal is another program, and closing it is on offer."""
        two = (TERMINAL, US,
               process(7032, "WindowsTerminal.exe", parent=13864, held=104,
                       windowed=True))

        self.assertEqual(Mib(85), held_by_this_window(two, US.pid))
        self.assertEqual([Mib(104)], [one.held for one in holders(two, US.pid)])


class ClosingAProgramAsksAllOfIt(unittest.TestCase):
    """Which process holds the window is the program's own business. Measured: Notepad
    keeps both its memory and its window in a child, and the list names the parent."""

    def test_the_process_the_list_names_is_in_it(self):
        self.assertIn(Pid(8108), family(DESKTOP, Pid(8108)))

    def test_every_process_of_the_program_is_in_it(self):
        self.assertEqual((Pid(8108), Pid(17040), Pid(34036)),
                         tuple(sorted(family(DESKTOP, Pid(8108)))))

    def test_nobody_elses_process_is_in_it(self):
        self.assertEqual((Pid(30780), Pid(33852), Pid(35772)),
                         tuple(sorted(family(DESKTOP, Pid(35772)))))

    def test_a_program_named_after_a_process_with_no_window_of_its_own(self):
        notepad = (process(25764, "explorer.exe"),
                   process(27236, "Notepad.exe", parent=25764),
                   process(28760, "Notepad.exe", parent=27236, held=52))
        named = [one.pid for one in holders(notepad, US.pid)]

        self.assertEqual([Pid(27236)], named)
        self.assertEqual((Pid(27236), Pid(28760)), family(notepad, Pid(27236)))

    def test_a_program_of_one_process_is_a_family_of_one(self):
        self.assertEqual((Pid(26284),), family(DESKTOP, Pid(26284)))

    def test_no_process_belongs_to_two_programs(self):
        found = [pid for one in holders(DESKTOP, US.pid)
                 for pid in family(DESKTOP, one.pid)]

        self.assertEqual(len(found), len(set(found)))


class WhatEnterDoesDependsOnTheRow(unittest.TestCase):
    """Asking is a message a program can refuse; ending is not. Which one a program
    gets is not a preference: measured, a hidden window does not answer the asking."""

    def marked(self, running, *pids):
        return closes(running, holders(running, US.pid), frozenset(pids))

    def test_a_program_with_a_window_up_is_asked_and_not_ended(self):
        given = self.marked(DESKTOP, Pid(35772))

        self.assertEqual(frozenset({Pid(35772), Pid(30780), Pid(33852)}), given.asked)
        self.assertEqual(frozenset(), given.ended)

    def test_a_program_with_nothing_on_screen_is_ended_and_not_asked(self):
        given = self.marked(DESKTOP, Pid(8108))

        self.assertEqual(frozenset(), given.asked)
        self.assertEqual(frozenset({Pid(8108), Pid(17040), Pid(34036)}), given.ended)

    def test_a_mark_on_something_that_is_no_longer_a_row_does_nothing(self):
        """A program can go between one pass and the next; a stale mark must not act."""
        given = self.marked(DESKTOP, Pid(99999))

        self.assertEqual(frozenset(), given.asked | given.ended)

    def test_what_is_not_marked_is_left_alone(self):
        self.assertEqual(frozenset(),
                         self.marked(DESKTOP).asked | self.marked(DESKTOP).ended)

    def test_both_kinds_at_once_go_to_their_own_side(self):
        given = self.marked(DESKTOP, Pid(35772), Pid(8108))

        self.assertEqual(frozenset({Pid(35772), Pid(30780), Pid(33852)}), given.asked)
        self.assertEqual(frozenset({Pid(8108), Pid(17040), Pid(34036)}), given.ended)

    def test_the_program_is_ended_before_the_processes_it_would_start_again(self):
        """A program whose first process is still running brings its helpers back."""
        self.assertEqual(Pid(8108), family(DESKTOP, Pid(8108))[0])


class WhatNobodyCanGiveBackIsNotOffered(unittest.TestCase):
    def test_the_compositor_is_left_out_of_it(self):
        """Its figure is the surfaces of the windows drawing through it, counted on
        their behalf and given back with them."""
        self.assertNotIn("dwm.exe", credited(DESKTOP))

    def test_what_the_compositor_holds_is_not_moved_onto_anybody_else(self):
        counted = sum(one.held for one in holders(DESKTOP, US.pid))

        self.assertEqual(Mib(262 + 208 + 170 + 59), Mib(counted))

    def test_the_router_is_never_proposed(self):
        """Unloading the model to load a model is not advice."""
        self.assertNotIn("llama-server.exe", credited(DESKTOP))


class TheProgramDoesNotOfferItsOwnWindow(unittest.TestCase):
    """Accepting would close the terminal it is drawn in, and there would be nobody
    left to see whether the memory came back."""

    def test_the_terminal_it_runs_in_is_not_on_the_list(self):
        self.assertNotIn("WindowsTerminal.exe", credited(DESKTOP))

    def test_what_that_terminal_holds_leaves_the_list_with_it(self):
        """Left off, not moved up to whatever started it."""
        without = sum(one.held for one in holders(DESKTOP, US.pid))
        counted = sum(one.held for one in holders(DESKTOP, ELSEWHERE))

        self.assertEqual(counted - TERMINAL.held, without)

    def test_run_from_somewhere_else_the_same_terminal_is_offered(self):
        self.assertIn("WindowsTerminal.exe", credited(DESKTOP, ours=ELSEWHERE))

    def test_the_shell_that_started_that_terminal_is_still_offered(self):
        """It stands above the terminal, and closing it closes no terminal."""
        self.assertEqual([Mib(170)], under(DESKTOP, 13864))

    def test_a_terminal_inside_an_editor_takes_both_off(self):
        """Closing either of them ends the program, so both are its own window."""
        nested = (process(1, "Zed.exe", held=139, windowed=True),
                  process(2, "WindowsTerminal.exe", parent=1, held=85, windowed=True),
                  process(3, "python.exe", parent=2))

        self.assertEqual((), holders(nested, Pid(3)))

    def test_another_window_of_the_same_program_is_still_offered(self):
        """It is this chain that is left off, not everything sharing its name."""
        two = (process(1, "WindowsTerminal.exe", held=85, windowed=True),
               process(2, "WindowsTerminal.exe", held=64, windowed=True),
               process(3, "python.exe", parent=1))

        self.assertEqual({"WindowsTerminal.exe": Mib(64)}, credited(two, ours=Pid(3)))

    def test_a_chain_of_windows_that_closes_on_itself_terminates(self):
        loop = (process(10, "a.exe", parent=11, held=50, windowed=True),
                process(11, "b.exe", parent=10, held=60, windowed=True))

        self.assertEqual((), holders(loop, Pid(10)))


class NothingIsCountedTwiceAndNothingLoops(unittest.TestCase):
    def test_the_total_credited_never_exceeds_the_total_held(self):
        held = sum(one.held for one in DESKTOP)

        self.assertLessEqual(sum(one.held for one in holders(DESKTOP, US.pid)), held)

    def test_a_parent_that_points_back_terminates(self):
        """Windows reuses process ids, so a parent chain can close on itself."""
        loop = (process(10, "a.exe", parent=11, held=50),
                process(11, "b.exe", parent=10, held=60))

        self.assertEqual({"b.exe": Mib(60), "a.exe": Mib(50)}, credited(loop))

    def test_a_chain_of_one_name_that_points_back_terminates(self):
        loop = (process(10, "a.exe", parent=11, held=50),
                process(11, "a.exe", parent=10, held=60))

        self.assertEqual(Mib(110), Mib(sum(one.held for one in holders(loop, US.pid))))

    def test_a_process_that_is_its_own_parent_terminates(self):
        self.assertEqual({"a.exe": Mib(50)},
                         credited((process(10, "a.exe", parent=10, held=50),)))

    def test_each_program_appears_once(self):
        found = [one.pid for one in holders(DESKTOP, US.pid)]

        self.assertEqual(len(found), len(set(found)))


class TheListReadsFromTheLargestDown(unittest.TestCase):
    def test_the_order_is_by_what_is_held(self):
        """The list is read to answer what to give up, and the answer is at that end."""
        given = [one.held for one in holders(DESKTOP, US.pid)]

        self.assertEqual(sorted(given, reverse=True), given)

    def test_the_same_input_gives_the_same_order(self):
        self.assertEqual(holders(DESKTOP, US.pid),
                         holders(tuple(reversed(DESKTOP)), US.pid))


class TheDesktopDrawsWhereItsCompositorHoldsMost(unittest.TestCase):
    """Which cards a desktop is on is asked of the compositor and of nothing else.

    Windows says nothing about it directly: over a remote session it detaches this
    machine's monitors, and the driver then reports none on any card while the desktop
    goes on holding what it holds. The compositor holds the most where it draws the
    desktop, and copies of windows rendered elsewhere where it does not.
    """

    def test_the_card_it_holds_the_most_on_draws_the_desktop(self):
        self.assertEqual((False, True), desktops((Mib(192), Mib(2490))))

    def test_a_card_holding_a_share_of_that_draws_one_too(self):
        """Monitors on two cards are two desktops, each wanting room to work at."""
        self.assertEqual((True, True), desktops((Mib(1200), Mib(2490))))

    def test_a_card_holding_less_than_the_share_holds_copies(self):
        """A window rendered on a card and composed onto a screen elsewhere is copied
        across, and the copy is credited to the compositor. Nobody works at that card."""
        under = Mib(int(Mib(2490) * DESKTOP_SHARE) - 1)

        self.assertEqual((False, True), desktops((under, Mib(2490))))

    def test_a_machine_nobody_is_logged_in_to_has_no_desktop_card(self):
        self.assertEqual((False, False), desktops((Mib(0), Mib(0))))

    def test_one_card_holding_it_all_draws_the_desktop(self):
        self.assertEqual((True,), desktops((Mib(3505),)))


if __name__ == "__main__":
    unittest.main()
