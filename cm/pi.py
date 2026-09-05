"""pi: point the pi agent on this machine at a router, and list what it serves.

Runs on the machine that calls the models rather than the one that holds them, so it
reads no card, no settings file and no preset. It asks the router what it serves and
believes the answer: the two cannot come to disagree, because there is only one of them.

The loop is here and it decides nothing. served.py reads the router's answer, policy.py
says what room pi's context-policy extension leaves, agent.py says what pi's two files
should hold, and this asks, writes and reports.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from . import agent, files, policy
from .agent import UnknownShape
from .refusal import Refusal
from .served import Router, Served, Unusable, read

DEFAULT_PORT = 18081
DEFAULT_PROVIDER = "llamacpp-cuda"
DEFAULT_BACKUPS = 10

# pi keeps both files here on every system it runs on.
CONFIG = Path(".pi") / "agent" / "models.json"
SETTINGS = "settings.json"

# Long enough for a router that is loading a model to answer, short enough that an
# address nobody is listening on comes back while a person is still watching.
TIMEOUT = 30


def main(argv: Sequence[str] | None = None) -> int:
    given = _arguments(argv if argv is not None else sys.argv[1:])

    try:
        _point(_router(given.server, given.port), given.config, given.provider,
               given.backups, given.preview)
    except (Refusal, UnknownShape) as refusal:
        print(refusal, file=sys.stderr)
        return 1

    return 0


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pi",
        description="Point the pi agent on this machine at the llama.cpp router, and "
                    "rewrite its model list from what the router actually serves.")
    parser.add_argument("server", metavar="host",
                        help="the machine running the router; a full URL is accepted "
                             "and reduced to host and port")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="router port, unless the address carries one")
    parser.add_argument("--config", type=Path, default=Path.home() / CONFIG,
                        help="pi's model configuration")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER,
                        help="what to call the provider, when none points there yet")
    parser.add_argument("--backups", type=int, default=DEFAULT_BACKUPS,
                        help="how many dated copies of pi's files to keep")
    parser.add_argument("--preview", action="store_true",
                        help="print what would be written, and write nothing")

    return parser.parse_args(list(argv))


def _point(router: Router, config: Path, provider: str, backups: int,
           preview: bool) -> None:
    print(f"Asking the router at {router.base_url} what it serves ...")

    models = _serving(router)
    for one in models:
        print(f"  {one.id:<24} {one.window:>7} tokens, reply up to {one.cap:>6}"
              f"   {one.name}")

    if not files.exists(config):
        raise Refusal(
            f"pi's model configuration was not found: {config}\n"
            "Install pi and start it once so it writes its configuration, then run this "
            "again. If it keeps its files elsewhere, pass --config.")

    listed = agent.pointed(_document(config), router, provider, models)

    settings = config.parent / SETTINGS
    held = _compaction(settings, models)

    print()
    if preview:
        _preview(config, listed, settings, held)
        return

    _save(config, listed, backups)
    _save(settings, held, backups)

    print("Check it inside pi with /context-policy: it flags a mismatch with a line "
          "starting '!'.")


def _compaction(settings: Path, models: Sequence[Served]) -> agent.Written:
    """pi's own two numbers, put behind the ones the extension will compute.

    The smallest pair across the models served is what pi has to fit behind, and it is
    printed: it is the number to compare with what /context-policy reports under the
    model with the shortest window.
    """
    document = _document(settings) if files.exists(settings) else {}

    match policy.room(models):
        case policy.Room() as room:
            print(f"The context-policy extension will hold back at least "
                  f"{room.reserve.most + 1} tokens and keep at most {room.keep.most}, "
                  "over these models. pi's own compaction has to stay behind that.")
            return agent.compacted(document, room)
        case policy.Unbounded():
            print("No model served has a window the context-policy extension will "
                  "govern, so pi's own compaction numbers are left as they are.")
            return agent.Written(document, ())


def _serving(router: Router) -> tuple[Served, ...]:
    """What the router serves, as models a client can be pointed at."""
    entries = _answer(router)
    if not entries:
        raise Refusal("The router answered but serves no models. Check its model "
                      "directory and the preset calibrate wrote.")

    models = []
    for entry in entries:
        match read(entry):
            case Served() as one:
                models.append(one)
            case Unusable(served, why):
                print(f"  ! {served or 'an entry'} left out: {why}")

    if not models:
        raise Refusal(f"None of the {len(entries)} models the router reports carried a "
                      "window, so there is nothing to write.")

    return tuple(sorted(models, key=lambda one: one.id))


def _answer(router: Router) -> tuple[object, ...]:
    """The router's model list. It is the source, so nothing is written without it."""
    try:
        with urllib.request.urlopen(router.models_url, timeout=TIMEOUT) as answer:
            reported = json.loads(answer.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as unreachable:
        raise Refusal(
            f"The router at {router.base_url} did not answer: {unreachable}\n"
            "It is where the model list comes from, so nothing was written. Check that "
            "it is running on that machine, that the port is open to this one, and that "
            "the address is right.") from None

    if not isinstance(reported, dict) or not isinstance(reported.get("data"), list):
        raise Refusal(f"{router.models_url} answered something that is not a model "
                      "list. Is that address a llama.cpp router?")

    return tuple(reported["data"])


def _document(path: Path) -> dict[str, object]:
    """One of pi's files, read. Its shape is pi's; only its syntax is checked here."""
    try:
        read_back = json.loads(files.read(path))
    except json.JSONDecodeError as unreadable:
        raise Refusal(f"{path} is not readable as JSON: {unreadable}") from None

    if not isinstance(read_back, dict):
        raise Refusal(f"{path} does not hold an object, so pi did not write it")

    return read_back


def _save(path: Path, written: agent.Written, backups: int) -> None:
    text = json.dumps(written.document, indent=2) + "\n"
    if files.exists(path) and files.read(path) == text:
        print(f"{path} already says this.")
        return

    kept = files.backup(path, backups) if files.exists(path) else path
    files.replace(path, text)

    print(f"Wrote {path}" + (f", previous version kept as {kept.name}"
                             if kept != path else ""))
    for change in written.changes:
        print(f"  {change}")


def _preview(config: Path, listed: agent.Written, settings: Path,
             held: agent.Written) -> None:
    for path, written in ((config, listed), (settings, held)):
        print(f"----- {path}, nothing written -----")
        print(json.dumps(written.document, indent=2))
        for change in written.changes:
            print(f"  {change}")
        print()


def _router(server: str, port: int) -> Router:
    """Where the router is, out of whatever was given for it.

    A pasted URL is the likeliest way this is given wrongly -- scheme, path and all --
    and the fix is mechanical, so it is done here rather than reported. A port in the
    address is the one meant: it was written more recently than the default.
    """
    named = server.strip().split("://")[-1].split("/")[0]

    host, _, written = named.partition(":")
    if written.isdigit():
        return Router(host, int(written))

    return Router(named, port)


if __name__ == "__main__":
    sys.exit(main())
