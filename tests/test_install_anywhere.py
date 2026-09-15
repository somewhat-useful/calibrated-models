"""Invariants of what installs where.

llama.cpp is installed from the builds its project publishes for Windows; pi is installed
from npm on whatever machine calls the router, which may run Windows, macOS or Linux. So
the command that installs pi has to start on a system with no Windows in it at all, and
the commands that install llama.cpp have to refuse there before they do anything.

Such a system is made in a child interpreter: the parts of ctypes that exist only on
Windows are taken away, and the platform renamed, before anything of this package is
imported. Nothing is installed and nothing is downloaded.
"""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WITHOUT_WINDOWS = """
import ctypes
import sys
for name in ("WinDLL", "windll", "OleDLL", "oledll", "WINFUNCTYPE", "WinError",
             "FormatError", "GetLastError", "get_last_error", "set_last_error", "HRESULT"):
    if hasattr(ctypes, name):
        delattr(ctypes, name)
sys.platform = "linux"
"""

# What stands in for everything that would install llama.cpp, so that a command going ahead
# says so instead of reaching the network.
GOING_AHEAD = """
from cm import engine
def went_ahead(*_, **__):
    raise AssertionError("went ahead")
engine.install = engine.release = went_ahead
"""


def elsewhere(program: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", WITHOUT_WINDOWS + program], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)


class PiInstallsOnAnySystem(unittest.TestCase):
    def test_the_command_that_installs_it_starts_with_no_windows_there(self):
        done = elsewhere("from cm import install\n"
                         "sys.exit(install.main(['pi', '--help']))\n")

        self.assertEqual(0, done.returncode, done.stderr)
        self.assertIn("--config", done.stdout)


class LlamaCppInstallsOnWindowsOnly(unittest.TestCase):
    def test_elsewhere_the_commands_that_install_it_refuse_before_anything(self):
        for command in (["llamacpp"], ["llamacpp", "--check"], ["slave"],
                        ["slave", "--print-command"]):
            with self.subTest(command=command):
                done = elsewhere(GOING_AHEAD + "from cm import install\n"
                                 f"sys.exit(install.main({command!r}))\n")

                self.assertEqual(1, done.returncode, done.stderr)
                self.assertIn("installs the Windows build of llama.cpp", done.stderr)
                self.assertNotIn("went ahead", done.stderr)
                self.assertEqual("", done.stdout)


if __name__ == "__main__":
    unittest.main()
