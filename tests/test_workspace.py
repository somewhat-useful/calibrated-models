"""Invariants of the working directory.

The directory the scripts were copied into holds everything: the code, the settings
file, the preset, the logs and the llama.cpp releases. Two of those are one machine's
and one person's, and the repository is public, so what keeps them out of it is checked
here rather than remembered.
"""

import unittest

from cm.workspace import ENGINES, TEMPLATE, engines, root, template

# What must never be committed: the releases, which are somebody else's build and
# hundreds of megabytes of it, and the settings file, which names this machine's models.
EXCLUDED = (f"{ENGINES}/", "settings.toml")


def ignored() -> tuple[str, ...]:
    """The rules the repository is excluded by, as written."""
    read = (root() / ".gitignore").read_text(encoding="utf-8").splitlines()
    return tuple(line.strip() for line in read
                 if line.strip() and not line.startswith("#"))


class TheWorkingDirectoryIsWhereTheScriptsAre(unittest.TestCase):
    def test_the_package_is_under_it(self):
        """Anchored to the code rather than to the directory a command was run from:
        the same release is meant whoever runs what from where."""
        self.assertTrue((root() / "cm" / "workspace.py").is_file())

    def test_the_releases_are_inside_it(self):
        self.assertEqual(root() / ENGINES, engines())
        self.assertEqual(root(), engines().parent)

    def test_the_engines_directory_is_hidden(self):
        """Dotted, so that what is in the working directory to be looked at stays
        visible beside what is in it because a program put it there."""
        self.assertTrue(ENGINES.startswith("."))


class NothingOfThisMachineEntersTheRepository(unittest.TestCase):
    def test_the_releases_and_the_settings_are_excluded(self):
        for rule in EXCLUDED:
            with self.subTest(rule=rule):
                self.assertIn(rule, ignored())

    def test_the_template_install_copies_is_not(self):
        """It is the one settings text the repository holds, and install has nothing to
        copy without it."""
        self.assertTrue(template().is_file())
        self.assertEqual(root() / TEMPLATE, template())
        self.assertNotIn(TEMPLATE, ignored())


if __name__ == "__main__":
    unittest.main()
