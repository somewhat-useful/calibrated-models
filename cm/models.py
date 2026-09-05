"""models: what the settings file asks the router to serve, and where each file came from.

Downloads nothing, on purpose. A repository is re-uploaded under the same file names,
so the file on disk and the one published now can differ without differing in size, and
a check by size would report a model as current when it is a build behind. What settles
it is the commit, and only a person can say whether the build they have is the build
they want -- so this puts in front of them what that decision takes: what the settings
file names, whether the file is here, the repository it came from, and where that
repository stands now. Fetching it is theirs.

The loop is here and it decides nothing. library.py says which repository a place in
the library names, and reads the project's answer about it; this asks and prints.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path, PurePath

from . import config, files, library, reading
from .config import Config, ConfigError, Model
from .library import Published, Reported, Repository, Revision, Unanswered, Unpublished
from .refusal import Refusal

# Hugging Face asks every caller to say what it is.
HEADERS = {"User-Agent": "llamacpp-local-library"}

# Long enough for the project to answer, short enough that a machine with no way out to
# the network reports it while a person is still watching.
TIMEOUT = 30

# What the project answers for a repository nobody outside can read. It does not
# distinguish a private one from one that was never there, and neither does this.
_NOT_SHOWN = (401, 403, 404)

_GIBIBYTE = 1073741824


def main(argv: Sequence[str] | None = None) -> int:
    given = _arguments(argv if argv is not None else sys.argv[1:])

    try:
        _show(given.settings)
    except (ConfigError, Refusal) as refusal:
        sys.stdout.flush()
        print(refusal, file=sys.stderr)
        return 1

    return 0


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="models",
        description="List the models the settings file names: whether the file is in "
                    "the library, which repository it came from, and what that "
                    "repository holds now. Downloads nothing.")
    parser.add_argument("--settings", type=Path, default=reading.DEFAULT,
                        help="the settings file to read")

    return parser.parse_args(list(argv))


def _show(settings: Path) -> None:
    read = reading.read(settings)

    print(f"Model library: {read.model_root}")

    asked: dict[Repository, Reported] = {}
    missing = []

    for model in read.models:
        if not _about(model, "", read, asked):
            missing.append(model.key)

    for model in read.withheld:
        _about(model, "   -- set aside, hidden = true", read, asked)

    _summary(read, tuple(missing), _unnamed(read))


def _about(model: Model, aside: str, read: Config,
           asked: dict[Repository, Reported]) -> bool:
    """One model as it stands. True where its file is in the library."""
    place = _place(model.path, read.model_root)
    here = files.exists(model.path)

    print()
    print(f"{model.key}{aside}")
    print(f"  file        {place}")
    print(f"  on disk     {_held(model.path) if here else 'not here'}")

    match library.origin(place):
        case Published() as published:
            print(f"  repository  {library.page_url(published)}")
            print(f"  revision    {_revision(published, asked)}")
        case Unpublished(where):
            print(f"  repository  nothing here can name it: {where} is not "
                  f"laid out as {library.LAYOUT}")

    return here


def _unnamed(read: Config) -> tuple[library.Held, ...]:
    """The models in the library that no entry names. Nothing is written for them here;
    that is scan's, which is where a person gets to look at a name before it is used."""
    named = read.models + read.withheld
    places = config.unnamed(named, files.under(read.model_root, library.SUFFIX),
                            read.model_root)

    return library.holds(places, frozenset(one.key for one in named))


def _summary(read: Config, missing: Sequence[str],
             unnamed: Sequence[library.Held]) -> None:
    print()
    print(f"{len(read.models) + len(read.withheld)} model(s) named, "
          f"{len(read.models) - len(missing)} of them in the library and offered.")

    if missing:
        print(f"Not here: {', '.join(missing)}")
        print()
        print(f"Fetch what is missing into {read.model_root}, keeping the "
              f"{library.LAYOUT} layout the entries name, then:")
        print("  python -m cm.calibrate")

    if unnamed:
        print()
        print(f"{len(unnamed)} model(s) in the library have no entry: "
              f"{', '.join(one.key for one in unnamed)}")
        print("  python -m cm.scan")


def _place(path: Path, root: Path) -> PurePath:
    """Where a model's file sits in the library, as the settings file put it there."""
    return path.relative_to(root) if path.is_relative_to(root) else path


def _held(path: Path) -> str:
    return f"{files.length(path) / _GIBIBYTE:.2f} GiB"


def _revision(published: Published, asked: dict[Repository, Reported]) -> str:
    """Where the repository stands now, asked once however many models it published."""
    if published.repository not in asked:
        asked[published.repository] = _asked(published)

    match asked[published.repository]:
        case Revision(commit, when):
            return f"{commit}  ({when:%Y-%m-%d})"
        case Unanswered(why):
            return f"not known -- {why}"


def _asked(published: Published) -> Reported:
    """What the project says about one repository, or why it did not say.

    A machine with no way out to the network still has a library and a settings file
    worth reporting, so this comes back as something to print rather than as an end to
    the run.
    """
    url = library.api_url(published)
    request = urllib.request.Request(url, headers=HEADERS)

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as answer:
            said = json.loads(answer.read().decode("utf-8"))
    except urllib.error.HTTPError as refused:
        if refused.code in _NOT_SHOWN:
            return Unanswered(f"{library.page_url(published)} is not a repository this "
                              "can see: private, gated, or no longer there")
        return Unanswered(f"{url} answered {refused}")
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as unreachable:
        return Unanswered(f"{url} did not answer: {unreachable}")

    return library.revision(said)


if __name__ == "__main__":
    sys.exit(main())
