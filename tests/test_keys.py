"""Invariants of what a key press means to the screen.

The console reports a press twice over: as the key that went down, and as the character
it produced. Only the first tells shift-tab from tab, because both of them produce a tab
character -- so what these check is that the naming is taken from the key, and that it
stays total over anything a keyboard can send.
"""

import unittest

from cm.screen import DOWN, MARK, NEXT, PREV, STAY, UP
from cm.vram import CLOSE, QUIT, named

TAB = 0x09
ENTER = 0x0D
SHIFT = 0x10
CONTROL = 0x11
CAPS_LOCK = 0x14
SPACE = 0x20
LEFT_ARROW = 0x25
UP_ARROW = 0x26
DOWN_ARROW = 0x28
LETTER_Q = 0x51
LETTER_R = 0x52
F5 = 0x74

# What the console puts in a record for a key that produces no character at all.
NOTHING = chr(0)


class TheTwoKeysThatWalkTheProfilesAreToldApart(unittest.TestCase):
    def test_tab_goes_on(self):
        self.assertEqual(NEXT, named(TAB, "\t", shift=False))

    def test_tab_with_shift_goes_back(self):
        self.assertEqual(PREV, named(TAB, "\t", shift=True))

    def test_the_character_is_the_same_for_both(self):
        """Which is why the key and not the character is what decides it."""
        self.assertNotEqual(named(TAB, "\t", shift=False),
                            named(TAB, "\t", shift=True))


class EveryKeyTheScreenActsOnArrivesUnderItsOwnName(unittest.TestCase):
    def test_enter_closes(self):
        self.assertEqual(CLOSE, named(ENTER, "\r", shift=False))

    def test_space_marks(self):
        self.assertEqual(MARK, named(SPACE, " ", shift=False))

    def test_the_arrows_walk_the_windows(self):
        self.assertEqual(UP, named(UP_ARROW, NOTHING, shift=False))
        self.assertEqual(DOWN, named(DOWN_ARROW, NOTHING, shift=False))

    def test_holding_shift_does_not_turn_them_into_something_else(self):
        """Only tab means a second thing with shift held."""
        for vk, char in ((ENTER, "\r"), (SPACE, " "), (UP_ARROW, NOTHING),
                         (DOWN_ARROW, NOTHING)):
            with self.subTest(vk=hex(vk)):
                self.assertEqual(named(vk, char, shift=False),
                                 named(vk, char, shift=True))


class ALetterArrivesAsItself(unittest.TestCase):
    def test_q_quits(self):
        self.assertEqual(QUIT, named(LETTER_Q, "q", shift=False))

    def test_a_capital_quits_too(self):
        """Caps lock is not a different intention."""
        self.assertEqual(QUIT, named(LETTER_Q, "Q", shift=True))

    def test_a_letter_the_screen_does_not_act_on_is_still_that_letter(self):
        self.assertEqual("r", named(LETTER_R, "r", shift=False))


class AnythingElseLeavesTheScreenWhereItStands(unittest.TestCase):
    def test_a_modifier_on_its_own_is_not_a_press(self):
        for vk in (SHIFT, CONTROL, CAPS_LOCK):
            with self.subTest(vk=hex(vk)):
                self.assertEqual(STAY, named(vk, NOTHING, shift=True))

    def test_a_key_that_produces_no_character_is_nothing_to_act_on(self):
        self.assertEqual(STAY, named(F5, NOTHING, shift=False))

    def test_an_arrow_this_screen_does_not_walk_is_nothing_to_act_on(self):
        """The lists are columns; there is nowhere to go sideways."""
        self.assertEqual(STAY, named(LEFT_ARROW, NOTHING, shift=False))


class EveryKeyOnAKeyboardIsNamedSomething(unittest.TestCase):
    def test_nothing_a_console_can_report_is_left_unanswered(self):
        for vk in range(0x100):
            for char in (NOTHING, "a", " ", "\t", "\r"):
                for shift in (False, True):
                    with self.subTest(vk=hex(vk), char=repr(char), shift=shift):
                        self.assertIsInstance(named(vk, char, shift), str)


if __name__ == "__main__":
    unittest.main()
