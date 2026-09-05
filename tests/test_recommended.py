"""Invariants of the recommendations kept in this repository.

The file is a claim about somebody else's model, so what is held down here is that a
claim cannot be made anonymously, cannot be written in a form that means two things, and
cannot quietly turn into a placement.
"""

import unittest
from typing import get_args

from cm import recommended, workspace
from cm.config import DERIVED, FLAGS
from cm.recommended import (Advice, Advised, Ambiguous, Recommended,
                            RecommendedError, Unknown)

CARD = "https://example.invalid/publisher/model"

# What a settings file refuses to carry, which is what scan hands the parser.
UNSETTABLE = DERIVED | FLAGS

ONE = f"""
[recommended.'qwen3.8-27b']
source = '{CARD}'

[recommended.'qwen3.8-27b'.settings]
temp  = '1.0'
top-k = '20'
"""


def parse(text: str):
    return recommended.parse(text, UNSETTABLE)


def refused(text: str) -> str:
    """The one line a recommendations file was refused with."""
    try:
        parse(text)
    except RecommendedError as error:
        return str(error)
    raise AssertionError("the file was accepted")


def rows(text: str):
    return parse(text)


class ARowSaysWhatToRunWithAndWhereThatCameFrom(unittest.TestCase):
    def test_a_file_naming_nothing_holds_no_rows(self):
        self.assertEqual((), parse("\n"))

    def test_the_settings_are_read_as_written(self):
        self.assertEqual({"temp": "1.0", "top-k": "20"}, dict(rows(ONE)[0].settings))

    def test_the_row_carries_where_it_was_read_from(self):
        self.assertEqual(CARD, rows(ONE)[0].source)

    def test_a_row_with_no_source_is_refused(self):
        """A number nobody can trace is one nobody can correct, which is the whole
        reason for keeping these in a repository rather than in a head."""
        text = ONE.replace(f"source = '{CARD}'\n", "")

        self.assertIn("source must be the address of the page", refused(text))

    def test_a_source_that_is_not_a_page_anybody_can_open_is_refused(self):
        """The field is for the page the numbers were read off and nothing else. A note
        about how the row came to be written is what must never sit there: this file
        goes to every machine, and one machine's working notes are no evidence and
        nobody else's business."""
        for written in ("", "the vendor said so", "copied from a settings file",
                        "huggingface.co/publisher/model"):
            with self.subTest(written=written):
                self.assertIn("source must be the address of the page",
                              refused(ONE.replace(CARD, written)))

    def test_a_value_left_unquoted_is_refused(self):
        """1.0 unquoted is a number, and a number read back is 1. A settings file and
        this one would then disagree about what the row says."""
        self.assertIn("temp must be quoted",
                      refused(ONE.replace("temp  = '1.0'", "temp  = 1.0")))

    def test_a_row_with_no_settings_is_refused(self):
        text = ONE.split("[recommended.'qwen3.8-27b'.settings]")[0]

        self.assertIn("settings is not a table", refused(text))

    def test_a_recommended_key_that_is_not_a_table(self):
        self.assertIn("recommended must be a table", refused("recommended = 'none'\n"))


class ARowCannotRecommendWhatASettingsFileWillNotHold(unittest.TestCase):
    """This file is written into settings files on machines that only pulled it. A row
    carrying something the reader refuses would go out to all of them and stop every
    command on every one, and the machine it was written on would be the last to know.
    So it is refused here, where one person sees it."""

    def refused_with(self, name: str) -> str:
        return refused(ONE.replace("temp  = '1.0'", f"{name} = '4096'"))

    def test_a_placement_calibrate_works_out_is_refused(self):
        self.assertIn("ctx-size is not a sampler value",
                      self.refused_with("ctx-size"))

    def test_every_key_the_reader_calls_derived_is_refused(self):
        for name in sorted(DERIVED):
            with self.subTest(name=name):
                self.assertIn(f"{name} is not a sampler value",
                              self.refused_with(name))

    def test_a_flag_of_the_entry_is_refused(self):
        """hidden and manual belong to the entry rather than to its sampler values, and
        under the settings header they do nothing at all."""
        for name in sorted(FLAGS):
            with self.subTest(name=name):
                self.assertIn(f"{name} is not a sampler value",
                              self.refused_with(name))


class OneModelGetsOneAnswer(unittest.TestCase):
    def written(self, *patterns: str) -> tuple:
        text = "".join(f"[recommended.'{one}']\nsource = '{CARD}'\n"
                       f"[recommended.'{one}'.settings]\ntemp = '1.0'\n\n"
                       for one in patterns)
        return parse(text)

    def test_a_model_no_row_covers_is_unknown_rather_than_guessed_at(self):
        match recommended.advice(self.written("gemma-4-12b"), "qwen3.8-27b"):
            case Unknown(stem):
                self.assertEqual("qwen3.8-27b", stem)
            case other:
                self.fail(f"answered {other}")

    def test_a_row_naming_the_model_answers_for_it(self):
        match recommended.advice(self.written("qwen3.8-27b"), "qwen3.8-27b"):
            case Recommended(_, _, pattern):
                self.assertEqual("qwen3.8-27b", pattern)
            case other:
                self.fail(f"answered {other}")

    def test_a_pattern_covers_a_family(self):
        match recommended.advice(self.written("qwen3.8-*"), "qwen3.8-27b"):
            case Recommended(_, _, pattern):
                self.assertEqual("qwen3.8-*", pattern)
            case other:
                self.fail(f"answered {other}")

    def test_the_row_written_for_one_release_beats_the_row_for_its_family(self):
        """Which is what makes a family row safe to write: a model that turns out to
        want something else takes a row of its own without the family being split."""
        match recommended.advice(self.written("qwen3.8-*", "qwen3.8-27b"),
                                 "qwen3.8-27b"):
            case Recommended(_, _, pattern):
                self.assertEqual("qwen3.8-27b", pattern)
            case other:
                self.fail(f"answered {other}")

    def test_two_rows_of_equal_reach_are_refused_rather_than_ranked(self):
        """Taking either would make the answer depend on the order they were typed."""
        match recommended.advice(self.written("qwen3.8-2*", "qwen3.8-*7"), "qwen3.8-27"):
            case Ambiguous(_, patterns):
                self.assertEqual({"qwen3.8-2*", "qwen3.8-*7"}, set(patterns))
            case other:
                self.fail(f"answered {other}")

    def test_a_pattern_is_matched_as_written_and_not_case_by_case(self):
        """A stem arrives lowercased, so a row written in capitals covers nothing and
        is better seen as covering nothing than as covering everything."""
        self.assertIsInstance(
            recommended.advice(self.written("Qwen3.8-27B"), "qwen3.8-27b"), Unknown)

    def test_a_row_naming_the_model_beats_a_longer_pattern_that_fits_it(self):
        """`qwen3.8-*7b` is one character longer than the name it happens to fit, and
        it is still a guess where the other row is the model itself."""
        match recommended.advice(self.written("qwen3.8-*7b", "qwen3.8-27b"),
                                 "qwen3.8-27b"):
            case Recommended(_, _, pattern):
                self.assertEqual("qwen3.8-27b", pattern)
            case other:
                self.fail(f"answered {other}")

    def test_the_pattern_that_is_more_of_a_name_wins(self):
        """Between two patterns it is how much of each is the model's own name, not how
        many characters they run to."""
        match recommended.advice(self.written("qwen3.8-*", "qwen3.8-27b-*"),
                                 "qwen3.8-27b-instruct"):
            case Recommended(_, _, pattern):
                self.assertEqual("qwen3.8-27b-*", pattern)
            case other:
                self.fail(f"answered {other}")

    def test_two_names_are_never_of_equal_reach(self):
        """Only one row can name a model outright, so an exact row is never one of two
        answers -- which is what makes writing one the way to settle an ambiguity."""
        match recommended.advice(self.written("qwen3.8-2*", "qwen3.8-*7",
                                              "qwen3.8-27"), "qwen3.8-27"):
            case Recommended(_, _, pattern):
                self.assertEqual("qwen3.8-27", pattern)
            case other:
                self.fail(f"answered {other}")

    def test_a_character_class_is_refused_rather_than_ranked(self):
        """[abc] is four characters standing for one, so counting them would rank it as
        four characters of a name that it does not have."""
        self.assertIn("a pattern may hold * and ?",
                      refused(ONE.replace("qwen3.8-27b", "qwen3.8-[23]7b")))


class EveryAnswerHasSomewhereToGo(unittest.TestCase):
    """scan writes an answer into a settings file, and only two of the three can be
    written: the third is a repository somebody has to correct. Nothing catches whatever
    else turns up, so a fourth answer added here has to be given a home there -- and
    this is what says so."""

    def test_the_one_answer_that_cannot_be_written_is_the_ambiguous_one(self):
        self.assertEqual(frozenset({Ambiguous}),
                         frozenset(get_args(Advice)) - frozenset(get_args(Advised)))

    def test_nothing_that_can_be_written_is_left_out_of_the_answers(self):
        self.assertEqual(frozenset(get_args(Advised)),
                         frozenset(get_args(Advice)) & frozenset(get_args(Advised)))


class TheFileInThisRepositoryIsWhatEveryMachineGets(unittest.TestCase):
    """Not a check of any number in it. What is held down is that scan can act on it.

    scan reads this file and writes what it says into a settings file, which is then
    read back by the same parser every command uses. A row this parser would refuse is
    a scan that leaves every machine's settings file unreadable.
    """

    def shipped(self):
        return parse(workspace.recommended().read_text(encoding="utf-8"))

    def test_it_parses_as_it_ships(self):
        self.assertNotEqual((), self.shipped())

    def test_every_row_names_a_page_its_numbers_can_be_checked_against(self):
        """And nothing else. Whoever reads a row has to be able to go and see whether
        the page still says that, without knowing anything about the machine the row
        was written on."""
        for row in self.shipped():
            with self.subTest(row=row.pattern):
                self.assertTrue(row.source.startswith("https://"), row.source)

    def test_no_row_writes_a_key_calibrate_works_out_itself(self):
        """A placement in an entry is refused by the reader, so one here would be a
        scan that writes a settings file nothing can read afterwards."""
        for row in self.shipped():
            with self.subTest(row=row.pattern):
                self.assertEqual(frozenset(), frozenset(row.settings) & DERIVED)

    def test_no_row_writes_a_flag_of_the_entry(self):
        """hidden and manual belong to the entry and not to its sampler values, and a
        row that wrote one would put it exactly where it does nothing."""
        for row in self.shipped():
            with self.subTest(row=row.pattern):
                self.assertEqual(frozenset(),
                                 frozenset(row.settings) & frozenset({"hidden",
                                                                      "manual"}))

    def test_every_row_answers_for_the_model_it_names(self):
        """A row shadowed by another is a row nobody can reach, and nothing in the file
        would say so."""
        shipped = self.shipped()

        for row in shipped:
            if "*" in row.pattern:
                continue
            with self.subTest(row=row.pattern):
                match recommended.advice(shipped, row.pattern):
                    case Recommended(_, _, pattern):
                        self.assertEqual(row.pattern, pattern)
                    case other:
                        self.fail(f"answered {other}")


if __name__ == "__main__":
    unittest.main()
