"""Invariants of the copy this keeps of pi's context-policy extension.

The rows below are the extension's own published table, not numbers this produced: a copy
checked against itself checks nothing. If a version of the extension disagrees with them,
these tests are what says so.
"""

import unittest

from cm.policy import (Bound, Policy, Room, Unbounded, Unpoliced, decided, room)
from cm.served import Served
from cm.units import Tokens

# window, reply cap, where the extension compacts, what it keeps verbatim.
PUBLISHED = ((8192, 2048, 5735, 2007),
             (16384, 4096, 11878, 4157),
             (32768, 8192, 23757, 8314),
             (65536, 8192, 56525, 19783),
             (65536, 16384, 47514, 16629),
             (131072, 16384, 113050, 39567),
             (131072, 32768, 95027, 33259),
             (262144, 32768, 226099, 48000))


def served(window, cap) -> Served:
    return Served(id=f"{window}-{cap}", name="a model", window=Tokens(window),
                  cap=Tokens(cap), modalities=("text",), reasoning=True)


# The profiles this machine's router serves. The shortest window carries the smallest
# reply cap, so it is the one everything else has to fit behind.
FLEET = (served(262144, 32768), served(131072, 32768), served(88000, 16384),
         served(52000, 16384), served(48000, 16384), served(28000, 8192))


class TheExtensionsOwnArithmetic(unittest.TestCase):
    def test_every_row_of_its_published_table(self):
        for window, cap, threshold, keep in PUBLISHED:
            with self.subTest(window=window, cap=cap):
                answer = decided(served(window, cap))

                self.assertEqual(Policy(reserve=Tokens(window - threshold),
                                        keep=Tokens(keep)),
                                 answer)

    def test_the_reserve_is_driven_by_the_cap_until_the_window_is_small(self):
        """Two models of one window, one allowed twice the reply, are not one case."""
        wide = decided(served(131072, 32768))
        narrow = decided(served(131072, 16384))

        self.assertGreater(wide.reserve, narrow.reserve)

    def test_a_window_with_no_room_left_for_a_tail_is_not_governed(self):
        """Below this the extension computes nothing and pi is on its own."""
        self.assertEqual(Unpoliced(Tokens(2900)), decided(served(2900, 1024)))


class WhatIsLeftForPiItself(unittest.TestCase):
    def test_the_smallest_of_the_fleet_is_what_binds(self):
        """8400 and 6860 are what /context-policy prints under the 28k profile."""
        self.assertEqual(Room(reserve=Bound(most=Tokens(8399), take=Tokens(8000)),
                              keep=Bound(most=Tokens(6860), take=Tokens(4000))),
                         room(FLEET))

    def test_the_reserve_allowed_is_one_under_the_policys(self):
        """The extension flags at 'at or above', so equal is already too much."""
        answer = room((served(28000, 8192),))

        self.assertEqual(Tokens(8399), answer.reserve.most)

    def test_what_is_kept_may_equal_the_policys(self):
        answer = room((served(28000, 8192),))

        self.assertEqual(decided(served(28000, 8192)).keep, answer.keep.most)

    def test_the_numbers_taken_are_round_and_inside_the_bounds(self):
        for one in FLEET:
            with self.subTest(model=one.id):
                answer = room((one,))

                self.assertLessEqual(answer.reserve.take, answer.reserve.most)
                self.assertLessEqual(answer.keep.take, answer.keep.most)
                self.assertEqual(0, answer.reserve.take % 500)
                self.assertEqual(0, answer.keep.take % 500)
                self.assertGreater(answer.keep.take, 0)

    def test_a_wider_model_added_to_the_fleet_changes_nothing(self):
        self.assertEqual(room(FLEET), room(FLEET + (served(262144, 32768),)))

    def test_a_fleet_none_of_which_is_governed(self):
        """Nothing computes a policy, so nothing here has anything to say about pi."""
        self.assertEqual(Unbounded(), room((served(2900, 1024), served(2048, 2048))))


if __name__ == "__main__":
    unittest.main()
