r"""scan: name the models in the library, and keep what they run with up to date.

A name is built from where a file sits and what it is called -- the model, the
quantisation of its weights, and the variant the repository directory carries where the
file name drops it -- and written into the settings file as an entry, along with the
sampler values published for that model. Those values come from recommended.toml in this
repository, so a correction is made once and reaches every machine that pulls it.

What happens to an entry is decided by the entry:

    no entry for the file        one is added, with what the repository recommends
    an entry                     its settings block is brought up to date
    an entry with manual         nothing at all
    one written some other way   nothing at all, and the run says which and why

So the file stays yours. Change a value and it stays changed only until the next run;
mark the entry manual = true and it stays changed for good. A model nothing in the
repository covers keeps whatever it has, and a new entry for one starts on neutral
values rather than on anybody's recommendation.

Taking a model out of service is not done by deleting its entry. The file would be
unnamed again and the next run would write it back; hidden = true leaves the entry in
place, which is what makes the removal stick.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath

from . import config, files, library, reading, recommended, workspace
from .config import ConfigError, Model, Writing
from .library import Key
from .recommended import Ambiguous, Recommended, RecommendedError, Row, Unknown


def main(argv: Sequence[str] | None = None) -> int:
    given = _arguments(argv if argv is not None else sys.argv[1:])

    try:
        _scan(given.settings)
    except (ConfigError, RecommendedError) as refusal:
        sys.stdout.flush()
        print(refusal, file=sys.stderr)
        return 1

    return 0


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scan",
        description="Add an entry to the settings file for every model in the library "
                    "that has none yet, and bring every entry's sampler values up to "
                    "date with what this repository recommends. An entry marked "
                    "manual = true is left alone.")
    parser.add_argument("--settings", type=Path, default=reading.DEFAULT,
                        help="the settings file to read and write")

    return parser.parse_args(list(argv))


@dataclass(frozen=True)
class _Left:
    """One entry scan did not touch, and the one line saying why."""

    key: str
    why: str


def _scan(settings: Path) -> None:
    # Read once and kept. What is written back is this text with entries edited into it,
    # so a second read would be a second file: anything that changed on disk between the
    # two would be read as settings and written back over.
    was = reading.text(settings)
    read = reading.parsed(was)
    if not files.exists(read.model_root):
        raise ConfigError(f"model_root does not exist: {read.model_root}")

    rows = recommended.parse(files.read(_shipped()), config.DERIVED | config.FLAGS)

    named = read.models + read.withheld
    fresh = config.unnamed(named, files.under(read.model_root, library.SUFFIX),
                           read.model_root)
    adding = tuple(_writing(rows, one.key, one.place)
                   for one in library.holds(fresh, frozenset(one.key for one in named)))
    updating, left = _existing(rows, named)

    print(f"Model library: {read.model_root}")
    print(f"Recommended:   {len(rows)} model(s) in {workspace.RECOMMENDED}")
    print()

    retuned = config.retuning(was, updating)
    text = config.naming(retuned.text, adding)
    if text != was:
        files.replace(settings, text)

    kept = frozenset(one.key for one in retuned.untouched)
    _reported(adding,
              tuple(one for one in updating if one.key not in kept),
              (*left, *(_Left(one.key, one.why) for one in retuned.untouched)))
    _closing(read.model_root, adding, fresh, changed=text != was)


def _shipped() -> Path:
    where = workspace.recommended()
    if not files.exists(where):
        raise RecommendedError(
            f"{workspace.RECOMMENDED} is not in this copy of the repository: {where}")

    return where


def _writing(rows: Sequence[Row], key: str, place: PurePath) -> Writing:
    """One entry as it should stand, with what the repository says about its model.

    Every case the repository can answer with is named. None is left to catch whatever
    else turns up: an answer this does not recognise, carried into an entry, is a
    settings block written out of nothing at all.
    """
    stem = library.stem(place)

    match recommended.advice(rows, stem):
        case Recommended() | Unknown() as advice:
            return Writing(key=Key(key), place=place, advice=advice)
        case Ambiguous(_, patterns):
            raise RecommendedError(
                f"{workspace.RECOMMENDED} covers {stem} with {len(patterns)} rows of "
                f"the same reach -- {', '.join(patterns)} -- so which of them applies "
                "would depend on the order they are written in. Make one of them "
                "narrower.")


def _existing(rows: Sequence[Row],
              named: Sequence[Model]) -> tuple[tuple[Writing, ...], tuple[_Left, ...]]:
    """The entries whose settings scan may bring up to date, and the ones it may not.

    An entry nothing covers is left as it is rather than reset: overwriting numbers
    somebody looked up with neutral ones would be worse than leaving them alone.
    """
    writings, left = [], []

    for one in named:
        if one.manual:
            left.append(_Left(one.key, "manual = true"))
            continue

        writing = _writing(rows, one.key, PurePath(one.path))
        match writing.advice:
            case Unknown(stem):
                left.append(_Left(one.key, f"nothing recommends {stem} yet"))
            case Recommended(_, _, _):
                writings.append(writing)

    return tuple(writings), tuple(left)


_DID = 12


def _reported(adding: Sequence[Writing], updating: Sequence[Writing],
              left: Sequence[_Left]) -> None:
    """What became of each entry, one line each."""
    for one in adding:
        print(f"  {'added':<{_DID}}{one.key}")
        print(f"  {'':<{_DID}}{one.place}")
        print(f"  {'':<{_DID}}{_from(one)}")

    for one in updating:
        print(f"  {'up to date':<{_DID}}{one.key}  <- {_from(one)}")

    for one in left:
        print(f"  {'left alone':<{_DID}}{one.key}  ({one.why})")


def _closing(model_root: Path, adding: Sequence[Writing],
             fresh: Sequence[PurePath], changed: bool) -> None:
    """What to do next, which is a different thing in each of four situations."""
    if adding:
        print()
        print(f"Added {len(adding)}. The name is what you will ask the router for, so "
              "look them over, then:")
        print("  python -m cm.calibrate")
        return

    if changed:
        print()
        print("Brought up to date with this repository. Then:")
        print("  python -m cm.calibrate")
        return

    if fresh:
        print(f"Nothing to add: the {len(fresh)} file(s) here that no entry names are "
              "projectors or prediction heads, which sit beside a model rather than "
              "being one.")
        return

    print("Every model in the library is named, and every entry says what this "
          "repository recommends.")
    print()
    print(f"Fetch what you want into {model_root}, keeping the {library.LAYOUT} "
          "layout, and run this again.")


def _from(writing: Writing) -> str:
    match writing.advice:
        case Recommended(_, _, pattern):
            return f"{workspace.RECOMMENDED}: {pattern}"
        case Unknown(stem):
            return f"neutral values -- nothing recommends {stem} yet"


if __name__ == "__main__":
    sys.exit(main())
