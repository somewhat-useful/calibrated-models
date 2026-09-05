"""This machine's settings file: where it is unless told otherwise, and reading it.

config.py is the other half and knows nothing about any machine -- text in, values out,
every way the text can be wrong ending in one line. Getting hold of the text, and
finding the library this machine keeps its weights in, is what has to happen first, and
every command does the same two steps in the same order before it does anything of its
own. They are here rather than written out in each.

The default is a name and not a path anywhere in particular: the file is looked for in
the directory the command was run from, which is what makes a copy of this repository
one working setup and running the commands from its root the way to use it.
"""

from pathlib import Path

from . import config, devices, files
from .config import Config, ConfigError

NAME = "settings.toml"
DEFAULT = Path(NAME)


def text(path: Path) -> str:
    """What the file says, refused with one line where the machine has none there."""
    if not files.exists(path):
        raise ConfigError(f"{path.name} not found at {path}")

    return files.read(path)


def parsed(said: str) -> Config:
    """Settings text, read against the library this machine keeps its weights in.

    Which library that is is a fact about the machine, so it is found here rather than
    in the parser: a settings file naming model_root does not use it, and one that
    leaves it out is asking for whatever LM Studio recorded on this machine.
    """
    return config.parse(said, devices.library())


def read(path: Path) -> Config:
    """The file, read. Both steps, for the commands that need no text of their own."""
    return parsed(text(path))
