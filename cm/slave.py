"""slave: start the llama.cpp worker that lends this machine's card to a router on another
machine, or stop it.

The loop is here and it decides nothing. rpc.py says what the worker is started with,
releases.py which release is the newest unpacked, and serving.py what is in the way of a
start; this starts, waits and reports.

Starting takes priority over a worker already on the port, for the same reason a router
start does: refusing would leave a person guessing which one is live. The scheduled
task goes with it, since ending the worker it started only makes it start another.
"""

import argparse
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from . import config, devices, files, proc, reading, releases, rpc, serving, session, task
from . import workspace
from .advise import Pid
from .config import ConfigError
from .machine import UnreadableDevice
from .place import Local, device_name
from .refusal import Refusal
from .serving import Occupied
from .units import Port

# How long to wait for a worker to go, or to start listening.
_WAITS = 60
_WAITED = 0.5

# How much of a worker's streams is worth showing after a start that did not survive.
_SHOWN = 40


def main(argv: Sequence[str] | None = None) -> int:
    given = _arguments(argv if argv is not None else sys.argv[1:])

    try:
        return given.act(given)
    except (ConfigError, Refusal, UnreadableDevice) as refusal:
        sys.stdout.flush()
        print(refusal, file=sys.stderr)
        return 1


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="slave",
        description="Start the llama.cpp worker that lends this machine's card to a router "
                    "on another machine, or stop it.")
    commands = parser.add_subparsers(required=True)

    start = commands.add_parser("start", help="start the worker")
    _common(start)
    start.add_argument("--print-command", action="store_true",
                       help="print what would be run, and start nothing")
    start.add_argument("--foreground", action="store_true",
                       help="run the worker in this console instead of detaching, so that "
                            "whatever started this stays alive with it")
    start.set_defaults(act=_start)

    stop = commands.add_parser("stop", help="stop the worker holding the port")
    _common(stop)
    stop.set_defaults(act=_stop)

    return parser.parse_args(list(argv))


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--settings", type=Path, default=reading.DEFAULT,
                        help="the settings file to read, where this machine has one")
    parser.add_argument("--port", type=int, default=rpc.DEFAULT_PORT,
                        help="the port the worker listens on")


def _start(given: argparse.Namespace) -> int:
    port = Port(given.port)
    worker = _worker(workspace.engines())
    card = _card()
    argv = rpc.arguments(port, card)

    if given.print_command:
        print("Preview only: --print-command starts nothing.")
        print(f"Executable: {worker}")
        print(f"Command:    {serving.written(worker, argv)}")
        return 0

    # A start that is the scheduled task leaves the task alone: ending it would end this.
    _take_over(port, leave_the_task=given.foreground)

    if given.foreground:
        print(f"Running the llama.cpp worker in this console on port {port}, lending {card}")
        return proc.attached((str(worker), *argv), worker.parent)

    settings: Path = given.settings
    logs = rpc.logs((settings.parent / _lending(settings).logs).resolve())
    files.ensure(logs.out.parent)

    started = proc.detached((str(worker), *argv), worker.parent, logs.out, logs.err)
    files.write(logs.pid, f"{started}\n")
    _listening(started, port, logs)

    print(f"Started the llama.cpp worker as process {started}, lending {card} on port "
          f"{port}")
    print(f"Log: {logs.err}")
    return 0


def _stop(given: argparse.Namespace) -> int:
    port = Port(given.port)
    stopped = _take_over(port, leave_the_task=False)

    print("The worker is stopped." if stopped else
          f"No worker was running, and nothing was holding port {port}.")
    return 0


def _lending(settings: Path) -> config.Lending:
    """What the settings file says, where this machine has one at all."""
    return config.lending(files.read(settings) if files.exists(settings) else "")


def _worker(root: Path) -> Path:
    """The worker of the newest release unpacked here, resolved at every start."""
    for release in releases.releases(files.directories(root)):
        worker = root / release.name / rpc.WORKER
        if files.exists(worker):
            return worker

    raise Refusal(f"No llama.cpp release under {root} carries {rpc.WORKER}.\n"
                  "Install one: python -m cm.install slave")


def _card() -> str:
    """The card lent: the one with the most memory, by the name llama.cpp gives it."""
    largest = max(devices.cards(), key=lambda one: one.card.total)
    return device_name(Local(largest.index, largest.card.total))


def _take_over(port: Port, leave_the_task: bool) -> tuple[Pid, ...]:
    """The worker holding this port, stopped, and the port confirmed free afterwards."""
    if not leave_the_task:
        _end_task()

    ours = Pid(os.getpid())
    asked = serving.in_the_way(Occupied(running=frozenset(session.processes(rpc.WORKER)),
                                        on_the_port=session.listening(port)), ours)
    for pid in asked:
        session.end(pid)

    left = _left(port, ours)
    if left:
        ids = ", ".join(str(pid) for pid in left)
        raise Refusal(f"Port {port} is still held by process {ids}. If it is not a worker "
                      "it was left alone deliberately; to end it, from an elevated "
                      f"PowerShell:\n\n  Stop-Process -Id {ids} -Force")

    for pid in asked:
        print(f"Stopped process {pid}, which was lending this card on port {port}")

    return asked


def _end_task() -> None:
    """The scheduled task, ended before the worker it is holding up. Asked by exit code,
    for the reason router.py gives."""
    if proc.run(("schtasks", "/Query", "/TN", task.WORKER_NAME)).code != 0:
        return

    print(f"Ending the scheduled task '{task.WORKER_NAME}' first: it would start another "
          "worker.")
    proc.run(("schtasks", "/End", "/TN", task.WORKER_NAME))


def _left(port: Port, ours: Pid) -> tuple[Pid, ...]:
    for _ in range(_WAITS):
        if not serving.still_held(session.listening(port), ours):
            return ()
        time.sleep(_WAITED)

    return serving.still_held(session.listening(port), ours)


def _listening(started: Pid, port: Port, logs: rpc.Logs) -> None:
    """Whether the worker came up on its port, which a detached one that exits in its first
    second would never say itself."""
    for _ in range(_WAITS):
        if started in session.listening(port):
            return
        if started not in session.processes(rpc.WORKER):
            break
        time.sleep(_WAITED)

    for path in (logs.err, logs.out):
        end = files.tail(path, _SHOWN)
        if end:
            print()
            print(f"Last lines of {path}")
            print(end)

    raise Refusal(f"{rpc.WORKER} did not start listening on port {port}. What it wrote is "
                  "above.")


if __name__ == "__main__":
    sys.exit(main())
