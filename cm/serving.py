"""How the router is run: one server, no model until a request names one.

The router starts holding nothing. A request naming a model in its `model` field makes
it load that model, keep at most `resident` of them, and unload after `idle` so the card
is free when nothing is being asked of it. Everything per model comes from the preset,
which is why no model, no window and no placement is named here.

Nothing is started and no file is opened. This says what the command line is and where a
running router writes; router.py runs it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .advise import Pid
from .units import Port, Seconds

# What the router is. The same executable a release is verified through, because a
# release that cannot say what build it is cannot be the one serving.
SERVER = "llama-server.exe"

# What the settings file need not say. The router listens on every address because the
# machine with the card exists to be called from others; one model at a time because a
# second resident model is video memory taken from the window of the first; a quarter of
# an hour because a model reloads in seconds and the card is wanted by whatever else the
# machine does.
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = Port(18081)
DEFAULT_RESIDENT = 1
DEFAULT_IDLE = Seconds(900)

# Where a router writes, under the log directory.
LOGS = "logs"


@dataclass(frozen=True)
class Serving:
    """How the router listens, what it keeps loaded, and where it writes."""

    host: str
    port: Port
    resident: int
    idle: Seconds
    logs: Path


@dataclass(frozen=True)
class Logs:
    """The files a running router leaves behind.

    Three of them because the streams say different things: llama.cpp's own log goes
    where it is told, and whatever the process writes before it has a log to write to --
    a missing library, a CUDA driver that will not initialise -- arrives on the two
    standard streams and would otherwise be lost with the window it was never shown in.
    """

    router: Path
    out: Path
    err: Path
    pid: Path


def logs(directory: Path) -> Logs:
    return Logs(router=directory / "router.log",
                out=directory / "router.stdout.log",
                err=directory / "router.stderr.log",
                pid=directory / "router.pid")


@dataclass(frozen=True)
class Occupied:
    """Two readings of the machine: what is running the server, and what is holding the
    port this router listens on.

    Kept apart rather than merged, because what they answer together is not what either
    answers alone. Neither is a reason to end a process by itself: a server on a port of
    its own is serving somebody, and a program on this port that is not a server is not
    this program's business. Where the two meet is the router that is in the way.
    """

    running: frozenset[Pid]
    on_the_port: frozenset[Pid]


def in_the_way(occupied: Occupied, ours: Pid) -> tuple[Pid, ...]:
    """The router to stop: the server that is holding this port. Both, not either.

    Running the server is not enough. Another copy of these scripts on a port of its
    own is nothing to do with this one, and a worker the server started runs the same
    executable and is its child -- ending everything by that name reaches past the
    router being replaced and into whatever else the machine was doing.

    Holding the port is not enough either. Something else listening there does stop a
    start, but it is somebody's program, and ending it is a decision a person makes
    rather than one a start makes on their behalf. It is reported instead.

    This process is never in it. It is the one thing that must survive the clearing --
    ending it would leave whatever it was told to do half done, and on a foreground
    start it is the router itself.
    """
    return tuple(sorted((occupied.running & occupied.on_the_port) - {ours}))


def still_held(on_the_port: frozenset[Pid], ours: Pid) -> tuple[Pid, ...]:
    """What has the port once the clearing is done, which had better be nothing.

    Asked of the port rather than of what was ended, because the question a start needs
    answered is whether it can bind -- not whether the processes aimed at are gone.
    """
    return tuple(sorted(on_the_port - {ours}))


def arguments(preset: Path, serving: Serving, log: Path) -> tuple[str, ...]:
    """The router's command line.

    No model and no sampler: naming one here would serve it under a configuration the
    preset does not describe, and the preset is what a client reads back to learn what
    it is talking to.
    """
    return ("--models-preset", str(preset),
            "--models-max", str(serving.resident),
            "--sleep-idle-seconds", str(serving.idle),
            "--host", serving.host,
            "--port", str(serving.port),
            "--no-ui",
            "--log-file", str(log))


def written(server: Path, argv: Sequence[str]) -> str:
    """The command line as a person would have to type it, quoting where they must."""
    return " ".join(_quoted(str(one)) for one in (server, *argv))


def _quoted(one: str) -> str:
    if not any(character in one for character in ' \t"'):
        return one
    return '"' + one.replace('"', '\\"') + '"'


def url(serving: Serving) -> str:
    """Where clients are pointed.

    0.0.0.0 is what the router binds rather than an address anything dials, and it is
    printed as it stands: replacing it with a name would be inventing one, and the
    person reading it knows what their machine is called.
    """
    return f"http://{serving.host}:{serving.port}/v1"
