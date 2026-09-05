"""Invariants of finding the library LM Studio keeps.

The two files are LM Studio's, written for its own purposes, and this only reads them.
So every way they can be absent, empty or unreadable has to end somewhere usable: at
the place LM Studio itself would have used.
"""

import json
import unittest
from pathlib import Path

from cm.lmstudio import home, models
from places import somewhere

# The profile LM Studio would be installed under is this machine's own, which is the
# one the program reads too.
PROFILE = Path.home()

POINTED_AT = somewhere("lmstudio")
DOWNLOADS = somewhere("models")

# What LM Studio writes into its settings file, with the keys that are not about
# storage left in: this is read out of somebody else's file, and it has company.
SETTINGS = json.dumps({"language": "en",
                       "downloadsFolder": str(DOWNLOADS),
                       "developerMode": True})


class TheHomeIsWhereThePointerSays(unittest.TestCase):
    def test_a_pointer_names_it(self):
        self.assertEqual(POINTED_AT, home(PROFILE, str(POINTED_AT)))

    def test_no_pointer_means_the_one_in_the_profile(self):
        """No file to read is not an error: LM Studio writes it only once moved."""
        self.assertEqual(PROFILE / ".lmstudio", home(PROFILE, ""))

    def test_the_line_is_taken_without_its_ending(self):
        self.assertEqual(POINTED_AT, home(PROFILE, f"{POINTED_AT}\r\n"))

    def test_a_pointer_holding_nothing_but_space_names_nothing(self):
        self.assertEqual(PROFILE / ".lmstudio", home(PROFILE, "  \n"))


class TheLibraryIsWhereTheSettingsSay(unittest.TestCase):
    HOME = PROFILE / ".lmstudio"

    def test_the_downloads_folder_is_the_library(self):
        self.assertEqual(DOWNLOADS, models(self.HOME, SETTINGS))

    def test_no_settings_at_all_means_the_default_beside_them(self):
        self.assertEqual(self.HOME / "models", models(self.HOME, ""))

    def test_settings_that_do_not_name_one(self):
        self.assertEqual(self.HOME / "models",
                         models(self.HOME, json.dumps({"language": "en"})))

    def test_settings_naming_it_something_that_is_not_a_path(self):
        for value in (None, 42, [], {}, "", "   "):
            with self.subTest(value=value):
                text = json.dumps({"downloadsFolder": value})

                self.assertEqual(self.HOME / "models", models(self.HOME, text))

    def test_a_settings_file_this_cannot_read(self):
        """Its file and its format. Nothing here is worth failing a calibration over."""
        for text in ("{", "null", "[1, 2]", "<html>", "\x00"):
            with self.subTest(text=text):
                self.assertEqual(self.HOME / "models", models(self.HOME, text))


class AWholeReadingOfAMachineThatMovedItsLibrary(unittest.TestCase):
    """Both files as they are on a machine where the library was moved to another disk."""

    def test_the_pointer_and_the_settings_lead_to_the_weights(self):
        where = home(PROFILE, str(PROFILE / ".lmstudio"))

        self.assertEqual(DOWNLOADS, models(where, SETTINGS))


if __name__ == "__main__":
    unittest.main()
