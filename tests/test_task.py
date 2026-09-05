"""Invariants of the task that starts the router at boot.

It is registered once and read years later by a scheduler nobody is watching, so what
matters is that the document says exactly what was meant: this account without a
password, at boot, restarted if it fails, running the interpreter in the foreground so
that the task lives as long as the server does.
"""

import unittest
from pathlib import Path
from xml.etree import ElementTree

from cm.task import Account, Runs, arguments, document
from places import somewhere

SCHEMA = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"

WHO = Account("BOX\\somebody")
SETTINGS = somewhere("calibrated models", "settings.toml")
STARTED = Runs(command=somewhere("python", "python.exe"),
               arguments=arguments(SETTINGS),
               working=somewhere("calibrated models"))


def read(started=STARTED, account=WHO) -> ElementTree.Element:
    """The document, back through a parser: what the scheduler will see, not the text."""
    return ElementTree.fromstring(document(started, account))


def one(task: ElementTree.Element, path: str) -> ElementTree.Element:
    found = task.find("/".join(f"{SCHEMA}{name}" for name in path.split("/")))
    if found is None:
        raise AssertionError(f"the document has no {path}")

    return found


class TheRouterIsStartedBySomethingThatOutlivesIt(unittest.TestCase):
    def test_the_interpreter_is_what_runs_not_a_server_binary(self):
        """The newest release is resolved at every start. A binary written into the
        task would be the one release that never changes again."""
        self.assertEqual(str(STARTED.command), one(read(), "Actions/Exec/Command").text)

    def test_it_is_told_to_stay_in_the_foreground(self):
        """Detached, the task would finish while the server it started went on, and a
        finished task is not restarted when the server fails."""
        self.assertIn("--foreground", one(read(), "Actions/Exec/Arguments").text)

    def test_it_runs_the_router_from_the_directory_it_was_registered_in(self):
        said = one(read(), "Actions/Exec/Arguments").text

        self.assertIn("-m cm.router start", said)
        self.assertEqual(str(STARTED.working),
                         one(read(), "Actions/Exec/WorkingDirectory").text)

    def test_the_settings_file_is_named_in_full(self):
        """Where a service reads its settings should not depend on a working directory
        somebody may change under it."""
        self.assertIn(f'"{SETTINGS}"', arguments(SETTINGS))


class NoPasswordIsStoredAnywhere(unittest.TestCase):
    def test_the_account_runs_it_without_one(self):
        self.assertEqual("S4U", one(read(), "Principals/Principal/LogonType").text)

    def test_under_the_account_it_was_given(self):
        self.assertEqual(WHO, one(read(), "Principals/Principal/UserId").text)

    def test_and_asks_for_no_elevation(self):
        """Absent is LeastPrivilege. A server that reads model files and binds a port
        has no use for more, and a task that runs elevated is one more that does."""
        self.assertIsNone(read().find(f"{SCHEMA}Principals/{SCHEMA}Principal/"
                                      f"{SCHEMA}RunLevel"))


class WhatTheSchedulerIsAskedFor(unittest.TestCase):
    def test_it_starts_at_boot(self):
        self.assertIsNotNone(one(read(), "Triggers/BootTrigger"))

    def test_a_router_that_fails_is_started_again(self):
        self.assertEqual("3", one(read(), "Settings/RestartOnFailure/Count").text)
        self.assertEqual("PT1M", one(read(), "Settings/RestartOnFailure/Interval").text)

    def test_it_is_never_stopped_for_running_too_long(self):
        """PT0S is no limit. This one is meant to run until the machine goes down."""
        self.assertEqual("PT0S", one(read(), "Settings/ExecutionTimeLimit").text)

    def test_a_second_one_is_not_started_beside_it(self):
        self.assertEqual("IgnoreNew", one(read(), "Settings/MultipleInstancesPolicy").text)

    def test_a_boot_it_was_not_up_for_is_caught_up_with(self):
        self.assertEqual("true", one(read(), "Settings/StartWhenAvailable").text)


class ADocumentIsWrittenNotAssembledByLuck(unittest.TestCase):
    def test_a_path_with_a_character_xml_reserves_comes_back_whole(self):
        awkward = Runs(command=somewhere("R&D", "python.exe"),
                       arguments=arguments(somewhere("R&D", "settings.toml")),
                       working=somewhere("R&D"))

        task = read(awkward)

        self.assertEqual(str(awkward.command), one(task, "Actions/Exec/Command").text)
        self.assertEqual(awkward.arguments, one(task, "Actions/Exec/Arguments").text)

    def test_an_account_with_one_too(self):
        self.assertEqual("BOX\\some & body",
                         one(read(account=Account("BOX\\some & body")),
                             "Principals/Principal/UserId").text)

    def test_it_declares_the_encoding_it_has_to_be_written_in(self):
        """The scheduler is handed a file, and a document that says UTF-16 while the
        file holds something else is not read at all."""
        self.assertIn('encoding="UTF-16"', document(STARTED, WHO).splitlines()[0])

    def test_the_command_is_a_path_not_a_line_to_be_split(self):
        spaced = Runs(command=Path("some place", "with spaces", "python.exe"),
                      arguments=STARTED.arguments, working=STARTED.working)

        self.assertEqual(str(spaced.command),
                         one(read(spaced), "Actions/Exec/Command").text)


if __name__ == "__main__":
    unittest.main()
