"""Where the weights are when the settings file does not name a place.

LM Studio keeps one library and writes down where it is, so this reads the record
rather than guessing: a pointer file in the home directory names LM Studio's own home,
and the settings file in that home names the folder it downloads into. Someone who
moved the library moved it there, and this follows them.

Text in, path out. Nothing is opened here, and every way the two files can be absent,
empty or unreadable ends at the place LM Studio itself would use.
"""

import json
from dataclasses import dataclass
from pathlib import Path

# Written beside the home directory, holding the path to LM Studio's home.
POINTER = ".lmstudio-home-pointer"

# Where that home is when no pointer names another, what it keeps its settings in, and
# the folder it downloads into unless the settings say otherwise.
HOME = ".lmstudio"
SETTINGS = "settings.json"
MODELS = "models"

# The setting naming that folder.
DOWNLOADS = "downloadsFolder"


@dataclass(frozen=True)
class Found:
    """A library that is there."""

    root: Path


@dataclass(frozen=True)
class Missing:
    """No library where LM Studio's own records point, and where that was."""

    looked: Path


Library = Found | Missing


def home(profile: Path, pointer: str) -> Path:
    """LM Studio's home: what the pointer names, or the default in the profile."""
    named = pointer.strip()
    return Path(named) if named else profile / HOME


def models(home: Path, settings: str) -> Path:
    """The folder it downloads weights into: what its settings name, or models/."""
    named = _downloads(settings)
    return Path(named) if named else home / MODELS


def _downloads(settings: str) -> str:
    """The folder named in LM Studio's settings, or "" where they name none.

    Its file and its format: settings this cannot read are settings that say nothing
    here, and the default stands. There is nothing to report, because there is nothing
    a person would do about it.
    """
    try:
        read = json.loads(settings)
    except json.JSONDecodeError:
        return ""

    if not isinstance(read, dict):
        return ""

    named = read.get(DOWNLOADS, "")
    return named.strip() if isinstance(named, str) else ""
