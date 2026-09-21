"""router: start the llama.cpp router, or stop whatever is serving.

The loop is here and it decides nothing. serving.py says what the command line is and
where a running router writes, releases.py says which release runs -- the build the
settings file records -- and this starts, waits and reports.

Starting takes priority over whatever is serving now. Refusing because something else
got there first would leave the person guessing which configuration is live, so a start
stops what is running -- the scheduled task included, since killing the router it
started only makes it start another.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from . import files, proc, reading, releases, serving, session, task, workspace
from .advise import Pid
from .config import ConfigError
from .machine import UnreadableDevice
from .refusal import Refusal
from .releases import Running
from .served import Router
from .serving import SERVER, Logs, Occupied, Serving
from .units import Port


# How long to wait for a signalled server to go, and with it the video memory it holds.
# Starting a replacement before the card is released is how a start fails to allocate.
_WAITS = 60
_WAITED = 0.5

# How long the router has to answer what it serves, once it is up. It has loaded no
# model at that point, so this is a listening socket answering, not an inference.
_ANSWER = 10

# How much of a log is worth showing after a start that did not survive.
_SHOWN = 40


def main(argv: Sequence[str] | None = None) -> int:
    given = _arguments(argv if argv is not None else sys.argv[1:])

    try:
        return given.act(given)
    except (ConfigError, Refusal, UnreadableDevice) as refusal:
        # What was reported on the way here first: the two streams are buffered
        # differently, and a refusal that arrives before the steps it followed reads as
        # a refusal to do any of them.
        sys.stdout.flush()
        print(refusal, file=sys.stderr)
        return 1


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="router",
        description="Start the llama.cpp router on this machine, or stop what is "
                    "serving. It holds no model until a request names one.")
    commands = parser.add_subparsers(required=True)

    start = commands.add_parser("start", help="start the router")
    start.add_argument("--settings", type=Path, default=reading.DEFAULT,
                       help="the settings file to read")
    start.add_argument("--print-command", action="store_true",
                       help="print what would be run, and start nothing")
    start.add_argument("--foreground", action="store_true",
                       help="run the server in this console instead of detaching, so "
                            "that whatever started this stays alive with it")
    start.add_argument("--startup-check-seconds", type=int, default=5,
                       help="how long to wait before reporting what it serves; 0 to "
                            "report nothing and return as soon as it is started")
    start.set_defaults(act=_start)

    stop = commands.add_parser("stop", help="stop whatever is serving")
    stop.add_argument("--settings", type=Path, default=reading.DEFAULT,
                      help="the settings file to read")
    stop.set_defaults(act=_stop)

    return parser.parse_args(list(argv))


def _start(given: argparse.Namespace) -> int:
    settings: Path = given.settings
    read = reading.read(settings)
    running = read.serving

    # Absolute, both of them: the server runs in the directory of the release it was
    # unpacked into, and a preset named relative to this one would be looked for there.
    preset = (settings.parent / read.preset_path).resolve()
    logs = serving.logs((settings.parent / running.logs).resolve())
    server = _server(workspace.engines(), read.build)
    argv = serving.arguments(preset, running, logs.router)

    if given.print_command:
        _preview(settings, server, preset, running, argv)
        return 0

    if not files.exists(preset):
        raise Refusal(f"The preset the router reads was not found: {preset}\n"
                      "It holds the placement worked out for this card, so the router "
                      "is not started without it. Write it: python -m cm.calibrate")

    files.ensure(logs.router.parent)

    # A start that *is* the scheduled task leaves the task alone: ending it here would
    # kill this process.
    _take_over(running.port, leave_the_task=given.foreground)

    if given.foreground:
        print(f"Running the llama.cpp router in this console on {serving.url(running)}")
        print(f"Binary: {server}")
        return proc.attached((str(server), *argv), server.parent)

    started = proc.detached((str(server), *argv), server.parent, logs.out, logs.err)
    files.write(logs.pid, f"{started}\n")

    print()
    print(f"Started the llama.cpp router as process {started}")
    print(f"Binary: {server}")
    print(f"URL: {serving.url(running)}")
    print(f"Models resident at once: {running.resident}, unloaded after {running.idle} "
          "s idle")
    print(f"Log: {logs.router}")

    if given.startup_check_seconds > 0:
        _survived(started, logs, running, given.startup_check_seconds)

    return 0


def _stop(given: argparse.Namespace) -> int:
    settings: Path = given.settings
    read = reading.read(settings)
    logs = serving.logs((settings.parent / read.serving.logs).resolve())

    stopped = _take_over(read.serving.port, leave_the_task=False)
    files.delete(logs.pid)

    print("The router is stopped." if stopped else
          f"Nothing was serving, and nothing was holding port {read.serving.port}.")
    return 0


def _server(root: Path, running: Running) -> Path:
    """What to run: the server of the build the settings file records, or of the newest
    release here where it records none.

    Resolved at every start, which is what makes an install take effect on the next
    start and nothing else.
    """
    for release in releases.to_run(releases.releases(files.directories(root)), running):
        server = root / release.name / SERVER
        if files.exists(server):
            return server

    raise Refusal(releases.unfound(root, SERVER, running))


def _preview(settings: Path, server: Path, preset: Path, running: Serving,
             argv: Sequence[str]) -> None:
    print("Preview only: --print-command starts nothing.")
    print(f"Executable: {server}")
    print(f"Settings:   {settings}")
    print(f"Preset:     {preset}")
    print(f"URL:        {serving.url(running)}")
    print(f"Command:    {serving.written(server, argv)}")


def _take_over(port: Port, leave_the_task: bool) -> tuple[Pid, ...]:
    """The router holding this port, stopped, and the port confirmed free afterwards.

    What is ended is what is both running the server and listening on the port. What
    the port is then checked for is anything at all: a start needs to bind it, and
    whether the thing in its way was aimed at makes no difference to that.
    """
    if not leave_the_task:
        _end_task()

    asked = serving.in_the_way(_occupied(port), _ours())
    for pid in asked:
        session.end(pid)

    left = _left(port)
    if left:
        raise Refusal(_unstoppable(left, port))

    for pid in asked:
        print(f"Stopped process {pid}, which was serving on port {port}")

    return asked


def _occupied(port: Port) -> Occupied:
    """What is running the server, and what is holding the port, read together."""
    return Occupied(running=frozenset(session.processes(SERVER)),
                    on_the_port=session.listening(port))


def _ours() -> Pid:
    """This process, which is never something to end: on a foreground start it is the
    router, and on any other it is what was told to do the clearing."""
    return Pid(os.getpid())


def _end_task() -> None:
    """The scheduled task, ended before the process it is holding up.

    Asked by exit code rather than by what it prints: schtasks reports a task's state in
    the language Windows was installed in, and a router left running because a word was
    matched in English is a start that then fails to allocate the card.
    """
    if proc.run(("schtasks", "/Query", "/TN", task.NAME)).code != 0:
        return

    print(f"Ending the scheduled task '{task.NAME}' first: it would start another "
          "router.")
    proc.run(("schtasks", "/End", "/TN", task.NAME))


def _left(port: Port) -> tuple[Pid, ...]:
    """What still has the port once whatever was ended has had time to go."""
    for _ in range(_WAITS):
        if not serving.still_held(session.listening(port), _ours()):
            return ()
        time.sleep(_WAITED)

    return serving.still_held(session.listening(port), _ours())


def _unstoppable(left: Sequence[Pid], port: Port) -> str:
    """A port still held is the one case where carrying on is worse than failing: the
    replacement would start, fail to bind, and leave a message about a socket that says
    nothing about what has it."""
    ids = ", ".join(str(pid) for pid in left)

    return (f"Port {port} is still held by process {ids}.\n\n"
            "Either it is a router this session may not signal -- one left behind by "
            "the scheduled task, running under another logon session -- or it is not a "
            f"router at all and was left alone deliberately: something that is not "
            f"{SERVER} is somebody else's program, and ending it is your call rather "
            "than this one's.\n\n"
            "To end it, from an elevated PowerShell:\n\n"
            f"  Stop-Process -Id {ids} -Force")


def _survived(started: Pid, logs: Logs, running: Serving, seconds: int) -> None:
    """Whether it is still there a moment later, and what it says it serves.

    A router that cannot allocate, cannot read the preset or cannot bind the port exits
    in the first seconds, and detached it exits unseen. Waiting is what turns that into
    a message instead of a silence.
    """
    time.sleep(seconds)

    if started not in session.processes(SERVER):
        _shown(logs)
        raise Refusal(f"{SERVER} exited during startup. What it wrote is above.")

    print(_answering(running))


def _shown(logs: Logs) -> None:
    for path in (logs.router, logs.err, logs.out):
        end = files.tail(path, _SHOWN)
        if end:
            print()
            print(f"Last lines of {path}")
            print(end)


def _answering(running: Serving) -> str:
    """What the router says it serves, asked over the loopback.

    Over 127.0.0.1 whatever it was told to bind: the question is whether this machine's
    router is up, and the address in the settings file may be one no client on this
    machine can dial.
    """
    router = Router("127.0.0.1", running.port)

    try:
        with urllib.request.urlopen(router.models_url, timeout=_ANSWER) as answer:
            reported = json.loads(answer.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as unanswered:
        return f"It is running but has not answered {router.models_url} yet: {unanswered}"

    listed = reported.get("data") if isinstance(reported, dict) else None
    if not isinstance(listed, list):
        return f"It is running but {router.models_url} answered no model list."

    served = tuple(str(one["id"]) for one in listed
                   if isinstance(one, dict) and "id" in one)

    return f"Serving {len(served)} model(s): " + ", ".join(served)


if __name__ == "__main__":
    sys.exit(main())
