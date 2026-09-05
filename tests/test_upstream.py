"""Invariants of choosing which published release to install.

The answers here are shaped like the ones the API gives, cut to the fields that are
read: a tag, when it was published, and the files hanging off it. What is being checked
is mostly the awkward middle -- a release tagged minutes ago whose archive is not up
yet, a release carrying every flavour but this machine's, a list holding tags from
another scheme entirely. The newest one that can actually be installed is rarely the
first one in the list.
"""

import unittest

from cm.units import Bytes
from cm.upstream import (Absent, Asset, Cuda, Present, Published, Unavailable,
                         binaries, directory, latest, published, runtime)

CUDA = Cuda("13.3")
OTHER = Cuda("12.4")

DOWNLOADS = "https://github.com/ggml-org/llama.cpp/releases/download"


def asset(name: str, size: int = 100) -> dict:
    """One published file, as the API reports it."""
    return {"name": name, "size": size, "browser_download_url": f"{DOWNLOADS}/{name}"}


def release(build: int, *assets: dict, when: str = "2026-09-01T09:00:00Z") -> dict:
    """One release, as the API reports it."""
    return {"tag_name": f"b{build}", "published_at": when, "prerelease": True,
            "assets": list(assets)}


def whole(build: int, cuda: Cuda = CUDA, **rest) -> dict:
    """A release with everything up: this machine's archive and the CUDA runtime."""
    return release(build,
                   asset(binaries(build, cuda), 402653184),
                   asset(runtime(cuda), 419430400),
                   asset(f"llama-b{build}-bin-ubuntu-x64.zip"),
                   **rest)


class AnArchiveIsNamedForTheBuildAndTheCudaItWasBuiltAgainst(unittest.TestCase):
    def test_the_windows_archive(self):
        self.assertEqual("llama-b10448-bin-win-cuda-13.3-x64.zip",
                         binaries(10448, CUDA))

    def test_the_runtime_is_named_for_the_cuda_alone(self):
        """The same file for every release of a CUDA version, which is why it can be
        carried over from the release already installed."""
        self.assertEqual("cudart-llama-bin-win-cuda-13.3-x64.zip", runtime(CUDA))

    def test_two_cuda_versions_are_two_different_archives(self):
        self.assertNotEqual(binaries(10448, CUDA), binaries(10448, OTHER))

    def test_what_it_is_called_once_it_is_unpacked_here(self):
        self.assertEqual("b10448-cuda13.3", directory(10448, CUDA))


class WhatWasPublishedIsReadOutOfWhatTheApiAnswers(unittest.TestCase):
    def test_the_newest_build_comes_first(self):
        given = published([release(10437), release(10448), release(9002)])

        self.assertEqual([10448, 10437, 9002], [one.build for one in given])

    def test_the_number_is_read_as_a_number_not_as_text(self):
        """Sorted as text, b9002 would come after b10448 and be taken for the newest."""
        given = published([release(9002), release(10448)])

        self.assertEqual(10448, given[0].build)

    def test_when_it_was_published_is_carried_over_for_a_person_to_read(self):
        given = published([release(10448, when="2026-08-31T22:14:07Z")])

        self.assertEqual("2026-08-31T22:14:07Z", given[0].when)

    def test_every_file_of_a_release_arrives_with_its_size_and_its_address(self):
        given = published([release(10448, asset("llama-b10448-bin-win-cuda-13.3-x64.zip",
                                                402653184))])

        self.assertEqual(
            (Asset(name="llama-b10448-bin-win-cuda-13.3-x64.zip",
                   size=Bytes(402653184),
                   url=f"{DOWNLOADS}/llama-b10448-bin-win-cuda-13.3-x64.zip"),),
            given[0].assets)

    def test_a_release_that_states_no_size_states_a_size_no_download_matches(self):
        """Nothing downloads to zero bytes, so a release whose sizes cannot be read
        refuses to install rather than installing something unmeasured."""
        unmeasured = {"name": binaries(10448, CUDA),
                      "browser_download_url": DOWNLOADS}

        given = published([release(10448, unmeasured)])

        self.assertEqual(Bytes(0), given[0].assets[0].size)


class WhatIsNotAPerBuildReleaseIsNotOne(unittest.TestCase):
    def test_another_tagging_scheme_is_passed_over(self):
        given = published([{"tag_name": "v0.1.0", "assets": []}, release(10448)])

        self.assertEqual([10448], [one.build for one in given])

    def test_a_tag_that_only_starts_like_one(self):
        for tag in ("b", "bx10448", "10448", "b10448-rc1"):
            with self.subTest(tag=tag):
                self.assertEqual((), published([{"tag_name": tag, "assets": []}]))

    def test_an_entry_that_is_not_a_release_at_all(self):
        self.assertEqual((), published(["something", 4, None]))

    def test_an_answer_that_is_not_a_list_of_releases(self):
        """A rate limit and an outage both answer with an object saying so."""
        self.assertEqual((), published({"message": "API rate limit exceeded"}))

    def test_nothing_published_is_not_an_error_here(self):
        self.assertEqual((), published([]))


class TheReleaseToInstallIsTheNewestThatCarriesTheArchive(unittest.TestCase):
    def test_the_newest_one_is_taken(self):
        given = latest(published([whole(10437), whole(10448)]), CUDA)

        self.assertEqual(10448, given.build)

    def test_the_archive_taken_is_the_one_for_this_machines_cuda(self):
        given = latest(published([whole(10448)]), CUDA)

        self.assertEqual("llama-b10448-bin-win-cuda-13.3-x64.zip", given.binaries.name)

    def test_a_release_carrying_every_flavour_but_this_one_is_not_an_update(self):
        given = latest(published([whole(10437, CUDA), whole(10448, OTHER)]), CUDA)

        self.assertEqual(10437, given.build)

    def test_the_order_of_the_answer_does_not_decide_it(self):
        given = latest(published([whole(10437), whole(10448), whole(9002)]), CUDA)

        self.assertEqual(10448, given.build)


class ANewerReleaseWhoseArchiveIsNotUpYetIsNotAnUpdateYet(unittest.TestCase):
    """A release's thirty files appear over several minutes. Minutes after a build is
    tagged, the newest tag exists with nothing under it worth downloading."""

    ANSWER = published([whole(10437), release(10448), release(10449)])

    def test_the_release_below_it_is_installed(self):
        self.assertEqual(10437, latest(self.ANSWER, CUDA).build)

    def test_the_ones_passed_over_are_named_newest_first(self):
        """Said out loud: stopping one build short is the ordinary case here, and a
        person watching the number ought to see why it is not the one they saw."""
        self.assertEqual((10449, 10448), latest(self.ANSWER, CUDA).incomplete)

    def test_nothing_is_passed_over_where_the_newest_is_complete(self):
        self.assertEqual((), latest(published([whole(10448)]), CUDA).incomplete)


class TheRuntimeArchiveIsReportedWhetherOrNotTheReleaseCarriesIt(unittest.TestCase):
    """It is carried over from the release already here, so a release without one is
    still installable. Whether it has to be downloaded is decided against what is
    installed, which is not something this knows."""

    def test_a_release_carrying_it(self):
        given = latest(published([whole(10448)]), CUDA)

        self.assertEqual(Present(Asset(name="cudart-llama-bin-win-cuda-13.3-x64.zip",
                                       size=Bytes(419430400),
                                       url=f"{DOWNLOADS}/cudart-llama-bin-win-cuda-"
                                           "13.3-x64.zip")),
                         given.runtime)

    def test_a_release_whose_binaries_are_up_and_whose_runtime_is_not(self):
        answer = published([release(10448, asset(binaries(10448, CUDA)))])

        self.assertEqual(Absent("cudart-llama-bin-win-cuda-13.3-x64.zip"),
                         latest(answer, CUDA).runtime)

    def test_the_runtime_of_another_cuda_version_is_not_this_one(self):
        answer = published([release(10448, asset(binaries(10448, CUDA)),
                                    asset(runtime(OTHER)))])

        self.assertEqual(Absent(runtime(CUDA)), latest(answer, CUDA).runtime)


class NothingToInstallSaysWhatWasLookedFor(unittest.TestCase):
    def refused(self, releases, cuda=CUDA) -> str:
        try:
            latest(releases, cuda)
        except Unavailable as refusal:
            return str(refusal)
        raise AssertionError("a release was chosen")

    def test_a_cuda_version_nothing_is_built_for_names_the_archive(self):
        said = self.refused(published([whole(10448)]), Cuda("9.1"))

        self.assertIn("llama-b<number>-bin-win-cuda-9.1-x64.zip", said)

    def test_it_says_which_settings_key_would_be_wrong(self):
        self.assertIn("cuda_version", self.refused(published([whole(10448)]),
                                                   Cuda("9.1")))

    def test_the_newest_tags_seen_are_shown_so_a_person_can_see_what_was_read(self):
        said = self.refused(published([release(one) for one in range(10440, 10450)]))

        self.assertIn("b10449, b10448, b10447, b10446, b10445", said)

    def test_an_answer_holding_no_releases_at_all(self):
        self.assertIn("none", self.refused(()))


class OnlyTheFieldsThatAreReadHaveToBeThere(unittest.TestCase):
    def test_a_release_with_no_assets_key(self):
        self.assertEqual((), published([{"tag_name": "b10448"}])[0].assets)

    def test_a_release_that_does_not_say_when(self):
        self.assertEqual("", published([{"tag_name": "b10448"}])[0].when)

    def test_a_release_built_by_hand_from_its_parts(self):
        """Nothing but a build number and one archive is enough to install from."""
        one = Published(build=10448, when="",
                        assets=(Asset(binaries(10448, CUDA), Bytes(1), DOWNLOADS),))

        self.assertEqual(10448, latest((one,), CUDA).build)


if __name__ == "__main__":
    unittest.main()
