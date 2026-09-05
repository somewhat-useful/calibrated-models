"""Invariants of the list of commands `python -m cm` prints.

A list of commands kept by hand beside the commands themselves is a list that goes wrong
quietly: a command added and not listed is one nobody finds, and a command listed and
not there is one somebody types and cannot run. Both are held down here.
"""

import contextlib
import importlib
import io
import pkgutil
import unittest

import cm
from cm import guide

# `python -m cm` itself, which prints the list rather than being on it.
ITSELF = "__main__"

# What a command's help calls the settings file.
SETTINGS = "--settings"


def helped(step: guide.Step) -> str:
    """What this step's command says for --help, asked of the command itself.

    A word in angle brackets is what the person puts something of their own in, so it
    is dropped: argparse is being asked what the command takes, not run.
    """
    words = step.command.split()
    module = importlib.import_module(f"cm.{words[0]}")
    asked = [one for one in words[1:] if not one.startswith("<")]

    written = io.StringIO()
    with contextlib.redirect_stdout(written), contextlib.suppress(SystemExit):
        module._arguments([*asked, "--help"])

    return written.getvalue()


def runnable() -> frozenset[str]:
    """Every module in the package that can be run: the ones that define main()."""
    found = set()
    for module in pkgutil.iter_modules(cm.__path__):
        if module.name == ITSELF:
            continue
        if hasattr(importlib.import_module(f"cm.{module.name}"), "main"):
            found.add(module.name)

    return frozenset(found)


class EveryCommandIsListedAndEveryListedCommandIsThere(unittest.TestCase):
    def test_nothing_runnable_is_left_off_the_list(self):
        """A command added to the package and not to the list is one that exists and
        that nobody is told about."""
        self.assertEqual(frozenset(), runnable() - guide.commands())

    def test_nothing_on_the_list_is_missing_from_the_package(self):
        """One renamed here and not there is a line that cannot be typed."""
        self.assertEqual(frozenset(), guide.commands() - runnable())


class ItReadsInATerminalWithoutWrapping(unittest.TestCase):
    """It is printed to a console and read there. A line that folds is a column that
    stops lining up, which is the whole reason for laying it out in columns."""

    def test_no_line_is_wider_than_a_narrow_terminal(self):
        for line in guide.path():
            with self.subTest(line=line):
                self.assertLessEqual(len(line), 88)

    def test_every_command_is_written_out_as_it_is_typed(self):
        written = "\n".join(guide.path())

        for name in guide.commands():
            with self.subTest(name=name):
                self.assertIn(f"{guide.RUN}.{name}", written)

    def test_the_stages_are_numbered_in_the_order_they_are_run(self):
        self.assertEqual([str(one) for one in range(1, len(guide.PATH) + 1)],
                         [stage.about.split(".")[0] for stage in guide.PATH])

    def test_what_is_not_part_of_the_path_is_not_numbered_into_it(self):
        """vram and router stop are run whenever they are wanted, and a number against
        them would say they belong somewhere in the sequence."""
        self.assertNotEqual((), guide.BESIDE)
        for one in guide.BESIDE:
            with self.subTest(one=one.command):
                self.assertFalse(one.command[0].isdigit())

    def test_emptying_one_of_the_lists_is_not_a_command_that_stops_working(self):
        """The columns are lined up against the widest command there is. With no
        commands there is no width, which is a layout rather than a failure."""
        self.assertEqual(0, guide._widest(()))


class WhatItSaysTheyTakeIsWhatTheyTake(unittest.TestCase):
    """The last line is one sentence about eleven commands, and a sentence like that
    goes stale silently: a command that grows --settings, or loses it, leaves the line
    saying something that was true once. So the line is built from a claim written
    against each command, and each claim is put to the command itself."""

    def test_every_command_that_is_said_to_take_settings_takes_it(self):
        for one in guide._every():
            if one.settings:
                with self.subTest(command=one.command):
                    self.assertIn(SETTINGS, helped(one))

    def test_every_command_that_is_said_not_to_takes_no_settings(self):
        for one in guide._every():
            if not one.settings:
                with self.subTest(command=one.command):
                    said = helped(one)
                    self.assertNotEqual("", said, "the command said nothing at all")
                    self.assertNotIn(SETTINGS, said)

    def test_the_sentence_names_the_ones_that_do_not(self):
        without = [one.command for one in guide._every() if not one.settings]

        self.assertNotEqual([], without)
        for command in without:
            with self.subTest(command=command):
                self.assertIn(command, guide.flags())

    def test_the_sentence_names_no_command_that_does(self):
        """Naming one that does is the way this line was wrong before: it said vram
        took no --settings, and vram takes one."""
        said = guide.flags()
        without = frozenset(one.command for one in guide._every() if not one.settings)

        for one in guide._every():
            if one.command not in without:
                with self.subTest(command=one.command):
                    self.assertNotIn(f" {one.command} ", f" {said} ")


if __name__ == "__main__":
    unittest.main()
