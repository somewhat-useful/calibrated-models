"""Invariants of the releases unpacked on this machine: which runs, which is kept.

Directory names in, decisions out. Nothing here looks at a disk, which is why the
awkward cases are cheap to state: a build number that sorts wrongly as text, an older
release someone unpacked this morning, a release a router is serving out of at the
moment it would be deleted.
"""

import unittest

from pathlib import Path

from cm.releases import (Pruned, Recorded, Release, Unrecorded, built_against, missing,
                         prune, releases, to_run, unfound, verifies)
from cm.upstream import Cuda


class AReleaseIsPickedByItsNumber(unittest.TestCase):
    def test_the_newest_comes_first(self):
        given = releases(("b10437-cuda13.3", "b10448-cuda13.3", "b9002-cuda12.4"))

        self.assertEqual(["b10448-cuda13.3", "b10437-cuda13.3", "b9002-cuda12.4"],
                         [one.name for one in given])

    def test_the_number_is_read_as_a_number_not_as_text(self):
        """Sorted as text, b9002 would come after b10448 and win."""
        given = releases(("b9002-cuda12.4", "b10448-cuda13.3"))

        self.assertEqual("b10448-cuda13.3", given[0].name)

    def test_what_it_was_built_against_does_not_order_it(self):
        given = releases(("b10448-cuda13.3", "b10449-cuda12.4"))

        self.assertEqual("b10449-cuda12.4", given[0].name)

    def test_a_release_needs_no_flavour_at_all(self):
        given = releases(("b10448",))

        self.assertEqual((Release("b10448", 10448),), given)


class WhatIsNotAReleaseIsNotOne(unittest.TestCase):
    """The directory they are unpacked in also holds logs, scripts and the rest."""

    def test_other_directories_are_passed_over(self):
        given = releases(("logs", "scripts", "distributed", "b10448-cuda13.3", ".staging"))

        self.assertEqual(["b10448-cuda13.3"], [one.name for one in given])

    def test_a_name_that_only_starts_like_one_is_not_one(self):
        self.assertEqual((), releases(("build", "b", "bx10448", "10448")))

    def test_nothing_installed_is_not_an_error(self):
        self.assertEqual((), releases(()))


class OlderReleasesArePrunedNewestFirst(unittest.TestCase):
    INSTALLED = releases(("b10430-cuda13.3", "b10437-cuda13.3", "b10448-cuda13.3"))

    def test_what_is_kept_is_the_newest_of_them(self):
        given = prune(self.INSTALLED, 2, ())

        self.assertEqual(["b10430-cuda13.3"], [one.name for one in given.remove])

    def test_keeping_more_than_there_are_removes_nothing(self):
        self.assertEqual(Pruned(remove=(), spared=()), prune(self.INSTALLED, 9, ()))

    def test_keeping_none_still_keeps_the_one_everything_runs(self):
        """A keep count of zero would delete the release the router is about to run."""
        given = prune(self.INSTALLED, 0, ())

        self.assertEqual(["b10437-cuda13.3", "b10430-cuda13.3"],
                         [one.name for one in given.remove])

    def test_the_order_they_arrive_in_does_not_decide_what_goes(self):
        given = prune(releases(("b10437-cuda13.3", "b10448-cuda13.3")), 1, ())

        self.assertEqual(["b10437-cuda13.3"], [one.name for one in given.remove])

    def test_nothing_installed_is_nothing_to_prune(self):
        self.assertEqual(Pruned(remove=(), spared=()), prune((), 2, ()))


class AReleaseAServerIsRunningFromIsNotRemoved(unittest.TestCase):
    """Removing it would pull the executable out from under a router that is serving,
    and the next model it loads is read from those files."""

    INSTALLED = releases(("b10430-cuda13.3", "b10437-cuda13.3", "b10448-cuda13.3"))

    def test_it_is_kept_although_it_is_past_keeping(self):
        given = prune(self.INSTALLED, 1, ("b10430-cuda13.3",))

        self.assertEqual(["b10430-cuda13.3"], [one.name for one in given.spared])

    def test_it_is_not_also_on_the_list_to_remove(self):
        given = prune(self.INSTALLED, 1, ("b10430-cuda13.3",))

        self.assertEqual(["b10437-cuda13.3"], [one.name for one in given.remove])

    def test_a_release_that_is_kept_anyway_is_not_reported_as_spared(self):
        """Only what would have gone is worth saying anything about."""
        given = prune(self.INSTALLED, 2, ("b10448-cuda13.3",))

        self.assertEqual((), given.spared)


class TheRuntimeLibrariesAreCarriedOverByName(unittest.TestCase):
    """Against the previous release rather than a list of the CUDA runtime's files
    written down here, so a library renamed or added is carried across all the same."""

    PREVIOUS = ("cudart64_13.dll", "cublas64_13.dll", "ggml-cuda.dll")

    def test_what_the_new_release_did_not_bring_is_carried(self):
        self.assertEqual(("cublas64_13.dll", "cudart64_13.dll"),
                         missing(("ggml-cuda.dll", "llama.dll"), self.PREVIOUS))

    def test_what_it_brought_itself_is_left_alone(self):
        """Its own build of a library, overwritten by the previous one, is the one
        failure this cannot be allowed to cause."""
        self.assertEqual((), missing(self.PREVIOUS, self.PREVIOUS))

    def test_the_two_spellings_of_a_name_are_one_name(self):
        self.assertEqual((), missing(("CUDART64_13.DLL",), ("cudart64_13.dll",)))

    def test_a_release_with_nothing_to_carry_over_from(self):
        self.assertEqual((), missing(("ggml-cuda.dll",), ()))


class AReleaseIsVerifiedBeforeItIsMovedIntoPlace(unittest.TestCase):
    SAID = "version: 0.1.0-dev (build 10448, commit ad1de39e0)\nbuilt with Clang 20.1.8"

    def test_the_build_it_says_it_is(self):
        self.assertTrue(verifies(self.SAID, 10448))

    def test_a_release_that_is_not_the_one_downloaded(self):
        self.assertFalse(verifies(self.SAID, 10437))

    def test_a_build_number_the_reported_one_merely_starts_with(self):
        """Build numbers run consecutively, so 1044 is the opening of 10448."""
        self.assertFalse(verifies(self.SAID, 1044))

    def test_a_server_that_printed_nothing(self):
        """What a half-extracted archive leaves: an executable that will not run."""
        self.assertFalse(verifies("", 10448))


class TheRuntimeIsCarriedOverOnlyFromAReleaseOfTheSameCuda(unittest.TestCase):
    """Its libraries carry the major version alone, so nothing in the files themselves
    tells a 13.3 runtime from a 13.4 one."""

    HERE = releases(("b11070-cuda13.4", "b10976-cuda13.3", "b10448", "b10900-cuda13.30"))

    def test_only_the_releases_built_against_that_version(self):
        self.assertEqual(["b11070-cuda13.4"],
                         [one.name for one in built_against(self.HERE, Cuda(13, 4))])

    def test_a_version_is_not_read_as_the_opening_of_another(self):
        self.assertEqual(["b10976-cuda13.3"],
                         [one.name for one in built_against(self.HERE, Cuda(13, 3))])

    def test_none_where_nothing_here_was_built_against_it(self):
        self.assertEqual((), built_against(self.HERE, Cuda(12, 4)))

class WhatRunsIsTheBuildTheSettingsFileRecords(unittest.TestCase):
    """A router and a slave run the same build or do not talk, so a newer release
    unpacked beside the recorded one changes nothing until install records it."""

    HERE = releases(("b11070-cuda13.4", "b11065-cuda13.4", "b10976-cuda13.3"))

    def test_the_recorded_build_and_no_other(self):
        self.assertEqual(["b11065-cuda13.4"],
                         [one.name for one in to_run(self.HERE, Recorded(11065))])

    def test_a_build_recorded_and_not_here_leaves_nothing_to_run(self):
        """Not the newest instead: that would be a machine quietly running another
        build than the one it was set to."""
        self.assertEqual((), to_run(self.HERE, Recorded(11080)))

    def test_nothing_recorded_runs_the_newest_here_as_before(self):
        self.assertEqual(["b11070-cuda13.4", "b11065-cuda13.4", "b10976-cuda13.3"],
                         [one.name for one in to_run(self.HERE, Unrecorded())])

    def test_a_recorded_build_missing_says_how_to_install_it(self):
        said = unfound(Path("engines"), "llama-server.exe", Recorded(11065))

        self.assertIn("records build 11065", said)
        self.assertIn("python -m cm.install llamacpp --build 11065", said)

    def test_nothing_here_says_how_to_install_something(self):
        said = unfound(Path("engines"), "llama-server.exe", Unrecorded())

        self.assertIn("carries llama-server.exe", said)
        self.assertIn("python -m cm.install llamacpp", said)


class AReleaseHeldIsNeverPastKeeping(unittest.TestCase):
    """The recorded build is what runs, and a build installed by naming it goes by
    hand."""

    INSTALLED = releases(("b11080-cuda13.4", "b11075-cuda13.4", "b11070-cuda13.4",
                          "b11065-cuda13.4"))

    def test_a_held_release_past_keeping_stays(self):
        pruned = prune(self.INSTALLED, 2, running=(), held={"b11065-cuda13.4"})

        self.assertEqual(["b11070-cuda13.4"], [one.name for one in pruned.remove])

    def test_it_is_not_said_to_have_a_server_running_from_it(self):
        pruned = prune(self.INSTALLED, 2, running=(), held={"b11065-cuda13.4"})

        self.assertEqual((), pruned.spared)

    def test_holding_one_within_keeping_removes_nothing_else(self):
        pruned = prune(self.INSTALLED, 2, running=(), held={"b11080-cuda13.4"})

        self.assertEqual(["b11070-cuda13.4", "b11065-cuda13.4"],
                         [one.name for one in pruned.remove])

if __name__ == "__main__":
    unittest.main()
