"""Invariants of deciding what to do about the context-policy extension.

The case that matters is the one that is invisible: a clone in pi's extensions
directory is loaded and reported nowhere, so a decision taken on pi's package list
alone would install a second copy of something already running, and the copy that then
answers is whichever pi finds first. Everything here is about looking at the disk first.
"""

import unittest

from cm.extension import (NAME, SOURCE, Cloned, Install, LeaveAlone, NotCloned,
                          Update, decided, loaded_from)
from places import somewhere

EXTENSIONS = somewhere("pi", "agent", "extensions")
WHERE = loaded_from(EXTENSIONS)

# What pi prints when asked what it carries, with and without this extension in it.
CARRIES = f"pi-coding-agent 1.4.0\n{NAME} 0.3.1\n"
CARRIES_NOTHING = "pi-coding-agent 1.4.0\n"


class SomebodysOwnCheckoutIsTheirs(unittest.TestCase):
    """A clone is how the extension is had by whoever edits it. Bringing it up to date
    is theirs to do, and installing a package beside it would take over what loads."""

    def test_a_checkout_is_left_where_it_is(self):
        self.assertEqual(LeaveAlone(WHERE), decided(Cloned(WHERE), CARRIES_NOTHING))

    def test_it_is_left_alone_even_where_pi_also_lists_the_package(self):
        self.assertEqual(LeaveAlone(WHERE), decided(Cloned(WHERE), CARRIES))

    def test_nothing_is_installed_over_it(self):
        self.assertNotIn(type(decided(Cloned(WHERE), CARRIES)), (Install, Update))


class EverythingElseIsPisOwnPackage(unittest.TestCase):
    def test_one_pi_already_carries_is_updated(self):
        self.assertEqual(Update(SOURCE), decided(NotCloned(), CARRIES))

    def test_one_it_does_not_carry_is_installed(self):
        self.assertEqual(Install(SOURCE), decided(NotCloned(), CARRIES_NOTHING))

    def test_a_pi_that_could_not_be_asked_installs_it(self):
        """Installing what is installed is what pi's own update does; leaving it out
        because a list could not be read is a client that then compacts on the wrong
        numbers."""
        self.assertEqual(Install(SOURCE), decided(NotCloned(), ""))


class TheCheckoutIsLookedForWherePiLoadsItFrom(unittest.TestCase):
    def test_under_the_extensions_directory_by_its_own_name(self):
        self.assertEqual(EXTENSIONS / NAME, loaded_from(EXTENSIONS))


if __name__ == "__main__":
    unittest.main()
