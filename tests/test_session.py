r"""Invariants of the command line an elevated run is handed.

Everything else in session.py reads the machine and is not testable without one. This is
not: it is text, and it is text that goes through two parsers -- cmd's and the program's
own -- with a UAC prompt in the middle. A mistake in it is a run that a person approved
and that then did something else.

The paths are taken from the system rather than written down, the way every test here
takes one. What each of them is for is in its name: one carries a space, one an
ampersand, one a double quote, and what the quoting rule does with each is the whole
subject.
"""

import unittest

from cm.session import (_Elevating, _parts, _shell_quoted, _through_cmd,
                        _unpassable)
from places import somewhere

# A space in each: a path under somebody's name usually has one, and it is what the
# quoting is there for.
PYTHON = somewhere("program files", "python.exe")
HERE = somewhere("calibrated models")
SAID = somewhere("cm-elevated-1.log")


def _elevating(*arguments: str) -> _Elevating:
    return _Elevating(command=PYTHON, arguments=arguments, working=HERE, said=SAID)


def handed(*arguments: str) -> str:
    return _through_cmd(_elevating(*arguments))


class TheElevatedRunStandsWhereThisOneDoes(unittest.TestCase):
    """The service that raises the prompt starts the elevated process in the system
    directory whatever it is told, so the directory is changed by the command itself.
    Without it `-m cm.autostart` runs from C:\\Windows\\System32 and finds no package.
    """

    def test_it_changes_to_the_directory_before_running_anything(self):
        given = handed("-m", "cm.autostart")

        self.assertLess(given.index("cd /d"), given.index("cm.autostart"))

    def test_it_changes_drive_as_well_as_directory(self):
        """A repository on another drive is not reached by a cd that stays on this
        one, and which drive either of them is on is not something this decides."""
        self.assertIn("cd /d", handed("-m", "cm.autostart"))

    def test_a_directory_that_is_gone_is_not_a_command_that_runs_anyway(self):
        """Joined so the run is conditional on the move, and bracketed so that what cmd
        says about the move lands in the same file as everything else."""
        given = handed("-m", "cm.autostart")

        self.assertTrue(given.startswith('/c "('), given)
        self.assertIn("&&", given)


class WhatItWroteComesBackWhateverStreamItUsed(unittest.TestCase):
    def test_both_streams_go_to_the_one_file(self):
        self.assertIn("2>&1", handed("-m", "cm.firewall"))

    def test_the_file_is_written_after_the_brackets_close(self):
        """Inside them the redirect would catch one command; outside it catches the
        group, which is the move and the run together."""
        given = handed("-m", "cm.firewall")

        self.assertLess(given.index(")"), given.index("2>&1"))


class EveryPathKeepsQuotesOfItsOwn(unittest.TestCase):
    """cmd strips the outermost pair from what follows /c and leaves the rest alone,
    which is the whole reason a command line like this can be built at all."""

    def test_the_outermost_pair_is_the_one_cmd_takes(self):
        given = handed("-m", "cm.autostart")

        self.assertTrue(given.startswith('/c "'), given)
        self.assertTrue(given.endswith('"'), given)

    def test_a_path_with_a_space_is_quoted(self):
        self.assertIn(f'"{PYTHON}"', handed())
        self.assertIn(f'"{HERE}"', handed())

    def test_a_word_with_nothing_awkward_in_it_is_left_alone(self):
        """Asked of the rule rather than of the whole line: whether the paths above
        need quotes depends on the directory this system hands out for scratch, and
        whether a plain word does not is the thing being said."""
        self.assertEqual("cm-elevated-1.log", _shell_quoted("cm-elevated-1.log"))

    def test_a_flag_is_left_alone(self):
        self.assertIn(" -m cm.autostart)", handed("-m", "cm.autostart"))

    def test_an_ampersand_in_a_directory_is_quoted_rather_than_run(self):
        """R&D is a directory name, and unquoted it is two commands."""
        awkward = _Elevating(command=PYTHON, arguments=("-m", "cm.autostart"),
                             working=somewhere("R&D", "models"), said=SAID)

        self.assertIn(f'"{somewhere("R&D", "models")}"', _through_cmd(awkward))

    def test_a_caret_is_quoted_rather_than_escaping_what_follows(self):
        self.assertEqual('"a^b"', _shell_quoted("a^b"))


class ARunThatCannotSurviveTheTripIsSaidToBeUnavailable(unittest.TestCase):
    """Escaping a double quote through cmd and then through the program's own parser is
    two conventions that disagree, so a run carrying one is refused. Refused as
    Unavailable rather than as an exception: the caller is left holding the rights it
    had, which is the same place Windows refusing the prompt leaves it, and its message
    about running the command elevated by hand covers both."""

    def test_a_quote_in_an_argument_is_found(self):
        self.assertEqual(('say "hello"',),
                         _unpassable(_elevating('say "hello"')))

    def test_a_quote_in_the_command_is_found(self):
        quoted = somewhere('a"b', "python.exe")
        awkward = _Elevating(command=quoted, arguments=(), working=HERE, said=SAID)

        self.assertEqual((str(quoted),), _unpassable(awkward))

    def test_a_quote_in_the_working_directory_is_found(self):
        quoted = somewhere('a"b')
        awkward = _Elevating(command=PYTHON, arguments=(), working=quoted, said=SAID)

        self.assertEqual((str(quoted),), _unpassable(awkward))

    def test_a_quote_in_the_file_it_writes_is_found(self):
        quoted = somewhere('a"b.log')
        awkward = _Elevating(command=PYTHON, arguments=(), working=HERE, said=quoted)

        self.assertEqual((str(quoted),), _unpassable(awkward))

    def test_an_ordinary_run_carries_nothing_unpassable(self):
        self.assertEqual((), _unpassable(_elevating("-m", "cm.autostart")))

    def test_everything_that_reaches_cmd_is_looked_at(self):
        """The check is worth only as much as its reach: what it looks at has to be
        every part the command line is built out of."""
        given = _elevating("-m", "cm.autostart")

        for part in _parts(given):
            with self.subTest(part=part):
                self.assertIn(part, _through_cmd(given))


if __name__ == "__main__":
    unittest.main()
