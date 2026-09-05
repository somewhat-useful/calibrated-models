"""autostart: have this machine start the router when it boots, or stop having it.

The loop is here and it decides nothing. task.py says what the scheduler is handed,
serving.py says what a running router writes and where clients reach it, and this checks
what it may do, hands the document to schtasks and reports what it said.

Every answer schtasks gives is taken from its exit code, never from what it printed:
proc.py says why.
"""

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from . import files, proc, reading, rights, serving, session, task, workspace
from .config import Config, ConfigError
from .machine import UnreadableDevice
from .refusal import Refusal
from .task import NAME, Account, Runs


def main(argv: Sequence[str] | None = None) -> int:
    given = _arguments(argv if argv is not None else sys.argv[1:])
    said = argv if argv is not None else sys.argv[1:]

    try:
        match _permitted(given, said):
            case session.Ran(code, wrote):
                print(wrote.rstrip())
                return code
            case rights.Held():
                pass

        if given.remove:
            _remove()
        else:
            _register(given)
    except (ConfigError, Refusal, UnreadableDevice) as refusal:
        sys.stdout.flush()
        print(refusal, file=sys.stderr)
        return 1

    return 0


def _permitted(given: argparse.Namespace,
               said: Sequence[str]) -> session.Ran | rights.Held:
    """The rights this run needs. What it needs them for is the scheduler: a task that
    runs at startup belongs to the machine rather than to whoever is logged in, and the
    scheduler says so by refusing.

    A preview asks for nothing. It reads no device and writes nothing, so a prompt in
    front of it would be a prompt for the rights to print.
    """
    if given.print_command:
        return rights.Held()

    match rights.asked(rights.Asking(module=__spec__.name, said=tuple(said),
                                     changing=_changing(given))):
        case rights.Refused(why):
            raise Refusal(why)
        case session.Ran(_, _) | rights.Held() as answer:
            return answer


def _changing(given: argparse.Namespace) -> str:
    return ("Removing what this machine runs at startup" if given.remove else
            "Registering what this machine runs at startup")


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="autostart",
        description=f"Register the scheduled task '{NAME}', which starts the llama.cpp "
                    "router when this machine boots, or remove it. Registering needs "
                    "an elevated session; nothing is stored anywhere, and no password "
                    "is asked for.")
    parser.add_argument("--settings", type=Path, default=reading.DEFAULT,
                        help="the settings file the task will name")
    parser.add_argument("--print-command", action="store_true",
                        help="print what would be registered, and register nothing")
    parser.add_argument("--start-now", action="store_true",
                        help="start the task once it is registered, instead of waiting "
                             "for the next boot")
    parser.add_argument("--remove", action="store_true",
                        help="remove the task, leaving the router to be started by hand")

    return parser.parse_args(list(argv))


def _register(given: argparse.Namespace) -> None:
    settings: Path = given.settings.resolve()
    read = reading.read(settings)

    started = Runs(command=Path(sys.executable),
                   arguments=task.arguments(settings),
                   working=workspace.root())
    document = task.document(started, Account(session.account()))

    if given.print_command:
        _preview(read, started, document)
        return

    preset = (settings.parent / read.preset_path).resolve()
    if not files.exists(preset):
        raise Refusal(f"The preset the router reads was not found: {preset}\n"
                      "A task registered without it starts a router that stops again. "
                      "Write it first: python -m cm.calibrate")

    # The task runs as this account without a logon session of its own, so it inherits
    # nothing that would create these on the way.
    files.ensure(serving.logs((settings.parent / read.serving.logs).resolve())
                 .router.parent)

    _handed(document)

    print(f"Registered the scheduled task '{NAME}'")
    print(f"Runs as:    {session.account()} (S4U, no password stored)")
    print(f"Runs:       {started.command} {started.arguments}")
    print(f"Working in: {started.working}")
    print(f"URL:        {serving.url(read.serving)}")

    if given.start_now:
        proc.asked(("schtasks", "/Run", "/TN", NAME), "start the task")
        print(f"Started '{NAME}'.")
        return

    print()
    print("It starts at the next boot. To start it now: python -m cm.autostart "
          "--start-now, or python -m cm.router start to run one in this session.")


def _remove() -> None:
    if proc.run(("schtasks", "/Query", "/TN", NAME)).code != 0:
        print(f"There is no scheduled task '{NAME}' on this machine.")
        return

    proc.asked(("schtasks", "/Delete", "/TN", NAME, "/F"), "remove the task")

    print(f"Removed the scheduled task '{NAME}'.")
    print("Nothing starts the router at boot now. A router already running is left "
          "running: python -m cm.router stop.")


def _handed(document: str) -> None:
    """The document, given to the scheduler through a file, which is how it takes one."""
    written = Path(tempfile.gettempdir()) / f"{NAME}.xml"

    try:
        files.write_utf16(written, document)
        proc.asked(("schtasks", "/Create", "/TN", NAME, "/XML", str(written), "/F"),
                   "register the task")
    finally:
        files.delete(written)


def _preview(read: Config, started: Runs, document: str) -> None:
    print("Preview only: --print-command registers nothing.")
    print(f"Task:       {NAME}")
    print(f"Runs as:    {session.account()} (S4U, no password stored)")
    print(f"Executable: {started.command}")
    print(f"Arguments:  {started.arguments}")
    print(f"Working in: {started.working}")
    print(f"URL:        {serving.url(read.serving)}")
    print()
    print(document)


if __name__ == "__main__":
    sys.exit(main())
