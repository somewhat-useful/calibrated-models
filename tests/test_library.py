"""Invariants of saying where a model in the library came from.

The settings file names a file by where it sits, and that place is also what says which
repository published it -- so the two can never disagree, there being only one of them.
Everything here is about that reading being the same wherever it is done, and about an
answer from the project being read rather than guessed at.
"""

import unittest
from datetime import datetime, timezone
from pathlib import PurePath, PurePosixPath, PureWindowsPath

from cm.library import (HOST, Held, Key, Published, Repository, Revision, Unanswered,
                        Unpublished, api_url, holds, origin, page_url, revision)

QWEN = Published(Repository("unsloth/Qwen3.8-27B-GGUF"))

# What the project answers about one repository, cut down to the two fields read.
ANSWER = {"id": "unsloth/Qwen3.8-27B-GGUF",
          "sha": "4ca720788d1e01f1bff70c033e0d0028fd02e502",
          "lastModified": "2026-08-20T12:04:25.000Z",
          "gated": False}


class WhereAFileCameFromIsWhereItSits(unittest.TestCase):
    def test_publisher_repository_file_names_the_repository(self):
        self.assertEqual(
            QWEN, origin(PurePath("unsloth", "Qwen3.8-27B-GGUF",
                                  "Qwen3.8-27B-UD-IQ4_XS.gguf")))

    def test_the_same_answer_whichever_separator_it_was_written_with(self):
        """The settings file is written with backslashes on the machine with the card.
        Which system reads it is not a fact about where the weights came from."""
        written = r"unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-IQ4_XS.gguf"

        self.assertEqual(QWEN, origin(PureWindowsPath(written)))
        self.assertEqual(QWEN, origin(PurePosixPath(written)))
        self.assertEqual(QWEN, origin(PurePosixPath(written.replace("\\", "/"))))

    def test_a_file_put_in_the_library_by_hand_names_nothing(self):
        alone = PurePath("qwen3.8.gguf")

        self.assertEqual(Unpublished(alone), origin(alone))

    def test_neither_does_one_a_directory_short(self):
        short = PurePath("Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ4_XS.gguf")

        self.assertEqual(Unpublished(short), origin(short))

    def test_nor_one_filed_deeper_than_the_layout(self):
        deeper = PurePath("mine", "unsloth", "Qwen3.8-27B-GGUF", "q.gguf")

        self.assertEqual(Unpublished(deeper), origin(deeper))


class ARepositoryIsNamedTwice(unittest.TestCase):
    def test_the_page_a_person_opens(self):
        self.assertEqual(f"{HOST}/unsloth/Qwen3.8-27B-GGUF", page_url(QWEN))

    def test_and_where_it_is_asked_about(self):
        self.assertEqual(f"{HOST}/api/models/unsloth/Qwen3.8-27B-GGUF", api_url(QWEN))


class WhatARepositoryHoldsNow(unittest.TestCase):
    def test_the_commit_and_when_it_moved(self):
        self.assertEqual(
            Revision("4ca720788d1e01f1bff70c033e0d0028fd02e502",
                     datetime(2026, 8, 20, 12, 4, 25, tzinfo=timezone.utc)),
            revision(ANSWER))

    def test_an_answer_that_is_not_a_record(self):
        self.assertIsInstance(revision(["a list of something"]), Unanswered)

    def test_one_that_names_no_commit(self):
        without = {key: value for key, value in ANSWER.items() if key != "sha"}

        self.assertIsInstance(revision(without), Unanswered)

    def test_one_that_does_not_say_when(self):
        without = {key: value for key, value in ANSWER.items() if key != "lastModified"}

        self.assertIsInstance(revision(without), Unanswered)

    def test_a_date_that_is_not_one_is_quoted_back(self):
        said = revision(dict(ANSWER, lastModified="last Tuesday"))

        self.assertIsInstance(said, Unanswered)
        self.assertIn("last Tuesday", said.why)


def at(*parts: str) -> PureWindowsPath:
    """A place in the library, written the way the library writes one."""
    return PureWindowsPath(*parts)


def named(*places: PureWindowsPath, taken: frozenset[Key] = frozenset()) -> tuple[str, ...]:
    """The names these files are held under, in the order they come back."""
    return tuple(one.key for one in holds(places, taken))


def only(*parts: str) -> str:
    """The name one file on its own is held under."""
    return named(at(*parts))[0]


class WhatAFileIsCalledIsWhatTheFileSays(unittest.TestCase):
    """The model and the quantisation of its weights, both off the file's own name.

    Which quantisation a person is talking to is not a detail: two of them are two
    different sets of answers at two different sizes, and the name they are asked for
    has to say which one arrived.
    """

    def test_the_model_and_the_quantisation_make_the_name(self):
        self.assertEqual("qwen3.8-27b-ud-iq4xs",
                         only("unsloth", "Qwen3.8-27B-GGUF",
                              "Qwen3.8-27B-UD-IQ4_XS.gguf"))

    def test_the_case_the_file_was_published_in_is_not_part_of_it(self):
        """A name is typed at a client, and typed again the next day."""
        self.assertEqual(only("p", "Gemma-4-12B-GGUF", "Gemma-4-12B-Q4_K_M.gguf"),
                         only("p", "gemma-4-12b-gguf", "gemma-4-12b-q4_k_m.gguf"))

    def test_the_underscores_of_a_quantisation_are_dropped(self):
        self.assertEqual("m-q4km", only("p", "m-GGUF", "m-Q4_K_M.gguf"))

    def test_a_layout_of_zero_says_nothing_and_is_dropped_with_them(self):
        """Q4_0 and Q8_0 carry no block layout, where Q4_K_M carries one worth naming.
        Keeping the 0 would put a digit in the name that distinguishes nothing."""
        self.assertEqual("m-q4", only("p", "m-GGUF", "m-Q4_0.gguf"))
        self.assertEqual("m-q8", only("p", "m-GGUF", "m-Q8_0.gguf"))

    def test_a_dynamic_quantisation_keeps_the_prefix_that_makes_it_one(self):
        """UD-IQ4_XS and IQ4_XS are quantised differently and answer differently."""
        self.assertNotEqual(only("p", "m-GGUF", "m-IQ4_XS.gguf"),
                            only("p", "m-GGUF", "m-UD-IQ4_XS.gguf"))

    def test_the_widths_that_are_not_quantisations_are_still_read(self):
        self.assertEqual("m-bf16", only("p", "m-GGUF", "m-BF16.gguf"))
        self.assertEqual("m-f16", only("p", "m-GGUF", "m-F16.gguf"))

    def test_a_file_whose_name_ends_in_no_quantisation_is_named_by_all_of_it(self):
        """A case rather than an error: somebody's own build is still a model."""
        self.assertEqual("my-own-build", only("p", "r", "my-own-build.gguf"))

    def test_two_quantisations_of_one_model_are_two_names(self):
        self.assertEqual(("qwen3.8-27b-q4km", "qwen3.8-27b-ud-iq4xs"),
                         named(at("a", "Qwen3.8-27B-GGUF", "Qwen3.8-27B-Q4_K_M.gguf"),
                               at("b", "Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ4_XS.gguf")))


class TheRepositoryIsTakenAtItsWordWhereItSaysMore(unittest.TestCase):
    """A repository directory often carries a variant the file name drops -- A3B,
    Instruct, Thinking. That variant is which model this is, not decoration."""

    def test_a_variant_the_file_name_drops_comes_from_the_directory(self):
        self.assertEqual("ornith-1.5-35b-a3b-q4km",
                         only("ornith-ai", "Ornith-1.5-35B-A3B-GGUF",
                              "Ornith-1.5-35B-Q4_K_M.gguf"))

    def test_a_directory_saying_the_same_adds_nothing(self):
        self.assertEqual("ornith-1.0-35b-q4km",
                         only("deepreinforce-ai", "Ornith-1.0-35B-GGUF",
                              "ornith-1.0-35b-Q4_K_M.gguf"))

    def test_a_directory_naming_something_else_is_not_believed(self):
        """It has to extend what the file says, not replace it: a directory somebody
        renamed would otherwise rename the model."""
        self.assertEqual("m-q4km", only("p", "a-folder-of-mine", "m-Q4_K_M.gguf"))


class WhatSitsBesideAModelIsNotOne(unittest.TestCase):
    def test_a_vision_projector_is_not_a_model(self):
        self.assertEqual((), named(at("p", "m-GGUF", "mmproj-m-BF16.gguf")))

    def test_a_prediction_head_published_on_its_own_is_not_a_model(self):
        """The head this serves is read out of the model's own file."""
        self.assertEqual((), named(at("p", "m-GGUF", "mtp-m-Q4_0.gguf")))

    def test_they_are_dropped_from_beside_a_model_that_is_kept(self):
        self.assertEqual(("m-q4km",),
                         named(at("p", "m-GGUF", "m-Q4_K_M.gguf"),
                               at("p", "m-GGUF", "mmproj-m-BF16.gguf"),
                               at("p", "m-GGUF", "mtp-m-Q4_0.gguf")))


class AModelInVolumesIsStillOneModel(unittest.TestCase):
    VOLUMES = (at("p", "m-GGUF", "m-Q4_K_M-00001-of-00003.gguf"),
               at("p", "m-GGUF", "m-Q4_K_M-00002-of-00003.gguf"),
               at("p", "m-GGUF", "m-Q4_K_M-00003-of-00003.gguf"))

    def test_only_the_first_volume_is_a_model(self):
        """llama.cpp opens the rest through it, so the others are not asked for."""
        self.assertEqual(1, len(named(*self.VOLUMES)))

    def test_the_volume_it_was_split_into_is_not_part_of_the_name(self):
        self.assertEqual(("m-q4km",), named(*self.VOLUMES))


class TwoFilesAreNeverCalledOneThing(unittest.TestCase):
    """A name that depended on which file was walked first is a name that moves."""

    SAME = (at("lmstudio-community", "Qwen3.8-27B-GGUF", "Qwen3.8-27B-Q4_K_M.gguf"),
            at("unsloth", "Qwen3.8-27B-GGUF", "Qwen3.8-27B-Q4_K_M.gguf"))

    def test_the_publisher_tells_apart_what_the_file_name_cannot(self):
        self.assertEqual(("lmstudio-community-qwen3.8-27b-q4km",
                          "unsloth-qwen3.8-27b-q4km"),
                         named(*self.SAME))

    def test_neither_of_them_keeps_the_short_name(self):
        """Both move. Letting one keep it would make the name a person types depend on
        which file was found first."""
        self.assertNotIn("qwen3.8-27b-q4km", named(*self.SAME))

    def test_a_name_already_spoken_for_is_stepped_around(self):
        """An entry a person renamed to this holds it, and scan writes something else
        rather than a second entry a table cannot hold twice."""
        one = at("unsloth", "Qwen3.8-27B-GGUF", "Qwen3.8-27B-Q4_K_M.gguf")

        self.assertEqual(("unsloth-qwen3.8-27b-q4km",),
                         named(one, taken=frozenset({Key("qwen3.8-27b-q4km")})))

    def test_a_file_the_publisher_cannot_tell_apart_falls_back_to_its_place(self):
        """Same publisher, same file name, two repositories: nothing shorter is left."""
        self.assertEqual(("p-one-m-q4km", "p-two-m-q4km"),
                         named(at("p", "one", "m-Q4_K_M.gguf"),
                               at("p", "two", "m-Q4_K_M.gguf")))


class EveryModelIsNamedAndTheOrderIsTheName(unittest.TestCase):
    def test_a_library_that_is_empty_holds_nothing(self):
        self.assertEqual((), named())

    def test_names_come_back_in_their_own_order_whatever_the_walk_found(self):
        """The order files arrive in is the file system's; the order they are written
        in should be the person's."""
        one = at("p", "b-GGUF", "b-Q4_K_M.gguf")
        two = at("p", "a-GGUF", "a-Q4_K_M.gguf")

        self.assertEqual(named(one, two), named(two, one))
        self.assertEqual(("a-q4km", "b-q4km"), named(one, two))

    def test_a_file_that_is_not_laid_out_as_the_library_lays_one_out_is_named_too(self):
        """Nothing can say where it came from, which does not stop it being served."""
        self.assertEqual(("m-q4km",), named(at("m-Q4_K_M.gguf")))

    def test_the_place_that_comes_back_is_the_place_that_went_in(self):
        place = at("unsloth", "Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ4_XS.gguf")

        self.assertEqual((Held(Key("qwen3.8-27b-ud-iq4xs"), place),),
                         holds((place,), frozenset()))


if __name__ == "__main__":
    unittest.main()
