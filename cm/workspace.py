"""Where this program was copied to, and what it keeps beside itself.

The directory holding the scripts is the working directory. The llama.cpp releases go
into .llamacpp inside it, the settings file and the preset sit beside them, and none of
that has to be written down anywhere: a copy of this repository is one working setup.

Anchored to this package rather than to the directory a command happened to be run
from. Running one from elsewhere still means this copy's releases, and a second copy of
the repository is a second setup with releases of its own.
"""

from pathlib import Path

# Where the releases are unpacked. Dotted and kept out of the repository: it holds
# several hundred megabytes of somebody else's build, and one machine's at that.
ENGINES = ".llamacpp"

# The settings file as it ships. install copies it to settings.toml on a machine that
# has none yet.
TEMPLATE = "settings.template.toml"

# What every machine's models are recommended to run with. In the repository rather
# than beside the settings file: the numbers belong to the models, so a correction made
# once reaches every machine that pulls it.
RECOMMENDED = "recommended.toml"


def root() -> Path:
    """The directory the scripts were copied into."""
    return Path(__file__).resolve().parent.parent


def engines() -> Path:
    """The directory the llama.cpp releases are unpacked into."""
    return root() / ENGINES


def template() -> Path:
    """The settings file as it ships, which is in the repository."""
    return root() / TEMPLATE


def recommended() -> Path:
    """What models are recommended to run with, which is in the repository."""
    return root() / RECOMMENDED
