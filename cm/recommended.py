"""What a model is recommended to be run with, kept in this repository rather than on
each machine.

A vendor publishes sampler values for its model, usually two sets: one in the chat
quickstart and one in the agentic and code benchmarks they report. The second is the one
this takes, because that is the work this router is for.

Those values are a property of the model, so they are the same on every machine that
holds the file, and keeping them here means a correction is made once and reaches every
machine that pulls it. What is particular to a machine -- which models it holds, what it
may spend on them -- stays in its own settings file, and anything written there wins.

A row is looked up by the model's stem: the file name with the quantisation left off. So
two quantisations of one model, and two publishers' repackagings of it, are one row.
Patterns hold a `*`, and the narrowest pattern that matches wins, which is what lets a
row for one release sit inside a row for its whole family. A row naming a model outright
wins over any pattern, however long: a name is the narrowest thing there is.

Nothing here is opened or fetched: text in, rows out, one row looked up at a time.
"""

import fnmatch
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# What a sampler value may be written as. Every one of them is a string: llama.cpp reads
# these off a command line, and a number here would be one more way for two files to
# disagree about whether 1.0 and 1 are the same value.
Setting = str


class RecommendedError(Exception):
    """A recommendations file that cannot be acted on, with the one line saying why."""


@dataclass(frozen=True)
class Row:
    """One model, or one family of them, and what it is recommended to run with."""

    pattern: str
    settings: Mapping[str, Setting]
    source: str


@dataclass(frozen=True)
class Recommended:
    """What a model should run with, where the numbers were read from, and which row
    answered -- which is what a person needs to find the row again to correct it."""

    settings: Mapping[str, Setting]
    source: str
    pattern: str


@dataclass(frozen=True)
class Unknown:
    """No row covers this model, so nobody has looked its numbers up yet."""

    stem: str


@dataclass(frozen=True)
class Ambiguous:
    """Two rows of equal reach cover this model, and nothing here can choose between
    them: which one won would depend on the order they happen to be written in."""

    stem: str
    patterns: tuple[str, ...]


Advice = Recommended | Unknown | Ambiguous

# What can be written into a settings file without anybody being asked anything. The
# third case cannot: two rows of the same reach is a repository somebody has to correct,
# and picking one of them here would hide that.
Advised = Recommended | Unknown

_TABLE = "recommended"

# What a source has to be: a page anybody can open. Held to it rather than left to
# judgement, because the one thing that must never end up in this field is a note about
# how a row came to be written -- this file goes to every machine, and where a row came
# from on one of them is nobody else's business and no evidence for anything.
_PAGE = "https://"

# What fnmatch reads as standing for something other than itself. A pattern is ranked by
# how much of it is a name rather than a wildcard, so these are the characters that do
# not count towards it.
_WILD = "*?"

# The one thing fnmatch reads specially that a pattern here may not hold. A character
# class would make the ranking a lie -- [abc] is four characters standing for one -- and
# nothing about a model's name calls for one.
_CLASS = "["


def parse(text: str, unsettable: frozenset[str]) -> tuple[Row, ...]:
    """The recommendations file as text, read into rows.

    The unsettable names are the ones a settings file refuses to carry, handed in
    because this file writes into that one: a row recommending a value that calibrate
    works out for itself would be written onto every machine that pulled this
    repository, and every command on all of them would then refuse to read its own
    settings file. So it is refused here, where one person sees it, rather than there.

    A TOMLDecodeError passes through: its message already says which line and why.
    """
    raw = tomllib.loads(text)

    rows = raw.get(_TABLE, {})
    if not isinstance(rows, dict):
        raise RecommendedError(
            f"{_TABLE} must be a table: one [{_TABLE}.\"model\"] entry per model")

    return tuple(_row(pattern, row, unsettable) for pattern, row in rows.items())


def advice(rows: Sequence[Row], stem: str) -> Advice:
    """What this model is recommended to run with, out of the rows that cover it.

    The narrowest pattern wins, so a row written for one release is taken over the row
    written for its family. Two of the same reach are not ranked: a file that says two
    different things about one model says nothing about it, and quietly taking either
    would make the answer depend on the order the rows were typed in.
    """
    covering = sorted((row for row in rows if fnmatch.fnmatchcase(stem, row.pattern)),
                      key=lambda row: _narrowness(row.pattern), reverse=True)

    match covering:
        case []:
            return Unknown(stem)
        case [one]:
            return Recommended(one.settings, one.source, one.pattern)
        case [one, second, *_] if _narrowness(one.pattern) > _narrowness(second.pattern):
            return Recommended(one.settings, one.source, one.pattern)
        case _:
            narrowest = _narrowness(covering[0].pattern)
            return Ambiguous(stem, tuple(row.pattern for row in covering
                                         if _narrowness(row.pattern) == narrowest))


def _narrowness(pattern: str) -> tuple[bool, int]:
    """How narrowly a pattern reaches, the more particular comparing greater.

    A name outright comes first, whatever its length: `ab*c` is four characters and
    `abc` three, but the model called abc is what the second row is about and the first
    is a guess that happens to fit. Among patterns it is how much of them is a name:
    `gemma-4-26b-*` says more about a file than `gemma-4-*` does.
    """
    return (not any(one in pattern for one in _WILD),
            sum(1 for one in pattern if one not in _WILD))


def _row(pattern: str, row: object, unsettable: frozenset[str]) -> Row:
    if not isinstance(row, dict):
        raise RecommendedError(f"{pattern}: not a table")

    if _CLASS in pattern:
        raise RecommendedError(
            f"{pattern}: a pattern may hold * and ? and nothing else that stands for "
            f"something other than itself. {_CLASS}abc] is four characters covering "
            "one, and what covers what is how these rows are ranked.")

    source = row.get("source")
    if not isinstance(source, str) or not source.startswith(_PAGE):
        raise RecommendedError(
            f"{pattern}: source must be the address of the page these numbers were read "
            f"off, starting {_PAGE}. A row that cannot be checked against its own "
            "source is a claim about somebody else's model with nothing behind it.")

    settings = row.get("settings")
    if not isinstance(settings, dict) or not settings:
        raise RecommendedError(f"{pattern}: settings is not a table of sampler values")

    return Row(pattern=pattern,
               settings=_settings(pattern, settings, unsettable),
               source=source)


def _settings(pattern: str, table: Mapping[str, object],
              unsettable: frozenset[str]) -> Mapping[str, Setting]:
    """The sampler values, all of them quoted, so that what is written here is what is
    passed on: 1.0 unquoted is a number, and a number read back is 1."""
    for name, value in table.items():
        if name in unsettable:
            raise RecommendedError(
                f"{pattern}: {name} is not a sampler value. A settings file refuses an "
                "entry carrying it, so a row recommending it would be written onto "
                "every machine that pulled this repository and stop every command on "
                "all of them.")
        if not isinstance(value, str):
            raise RecommendedError(
                f"{pattern}: {name} must be quoted -- '1.0' rather than 1.0, so that "
                "this file and a settings file cannot disagree about what it says.")

    return {name: str(value) for name, value in table.items()}
