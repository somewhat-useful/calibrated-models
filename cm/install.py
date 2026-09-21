"""install: put on this machine what this router is made of.

Four things, and they belong to different machines. `llamacpp` is the engine, on the
machine with the router and on a machine lending its card alike: the router runs one
program of a release and the worker another, and the two have to be the same build.
`pi` is the agent that calls the router, on any machine that does -- including that
one, where somebody works at the card as well as serving from it. `slave` makes a
machine that has llama.cpp lend its card, and `master` names that machine in the router
machine's settings file, so that calibrate places models across its card too.

The loop is here and it decides nothing: engine.py installs a release, client.py
installs pi and its extension, task.py and lan.py say what a slave registers and admits,
config.py how a settings file names a slave, and this reads the command line and reports
a refusal.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import (autostart, client, config, devices, engine, files, lan, proc, reach,
               reading, rights, rpc, session, slave, task, upstream, workspace)
from .config import ConfigError
from .machine import UnreadableDevice
from .pi import CONFIG
from .place import DEFAULT_SLAVE_RESERVE, Endpoint, Worker
from .refusal import Refusal
from .units import Mib, Port


def main(argv: Sequence[str] | None = None) -> int:
    said = list(argv if argv is not None else sys.argv[1:])
    given = _arguments(said)

    try:
        return given.act(given, said)
    except (ConfigError, Refusal, UnreadableDevice,
            upstream.Unavailable) as refusal:
        sys.stdout.flush()
        print(refusal, file=sys.stderr)
        return 1


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="install",
        description="Install what this machine is missing, or bring it up to date.")
    what = parser.add_subparsers(required=True)

    llamacpp = what.add_parser(
        "llamacpp", help="the llama.cpp release the router or the worker runs",
        description="Install the newest llama.cpp release published for the CUDA "
                    "version this machine's driver runs, or the build named, record it "
                    "in the settings file as the build to run, and remove the releases "
                    "past keeping.")
    llamacpp.add_argument("--settings", type=Path, default=reading.DEFAULT,
                          help="the settings file to read and record the build in")
    llamacpp.add_argument("--check", action="store_true",
                          help="say which build would be installed, and download and "
                               "record nothing")
    llamacpp.add_argument("--build", type=int, metavar="NUMBER",
                          help="install this build rather than the newest: the one the "
                               "other machine's --check printed, so that the router and "
                               "the slave run the same. Pruning leaves it alone")
    llamacpp.add_argument("--force", action="store_true",
                          help="download even where the release is already here, after a "
                               "partial or damaged install")
    llamacpp.set_defaults(act=_llamacpp)

    pi = what.add_parser(
        "pi", help="the pi agent and the context-policy extension",
        description="Install or update the pi agent and the context-policy extension "
                    "on this machine. What points it at a router is cm.pi.")
    pi.add_argument("--config", type=Path, default=Path.home() / CONFIG,
                    help="pi's model configuration")
    pi.set_defaults(act=_pi)

    slave = what.add_parser(
        "slave", help="lend this machine's card to a router on another machine",
        description="Start the worker that lends this machine's card at every boot, "
                    "from the release install llamacpp put here, and admit the local "
                    "subnet to its port. Needs an elevated session; stores no password.")
    slave.add_argument("--settings", type=Path, default=reading.DEFAULT,
                       help="the settings file to read, where this machine has one")
    slave.add_argument("--port", type=int, default=rpc.DEFAULT_PORT,
                       help="the port the worker listens on")
    slave.add_argument("--print-command", action="store_true",
                       help="print what would be registered, and change nothing")
    slave.add_argument("--remove", action="store_true",
                       help="stop starting the worker at boot and close its port, "
                            "leaving llama.cpp installed")
    slave.set_defaults(act=_slave)

    master = what.add_parser(
        "master", help="name the slave whose card this machine's router may use",
        description="Write the slave into this machine's settings file: where its worker "
                    "listens and how much memory its card has. calibrate then places "
                    "models across its card as well as this machine's own.")
    master.add_argument("address", nargs="?",
                        help="the slave's host, or host:port")
    master.add_argument("memory", nargs="?",
                        help="its card's memory in gibibytes: 12, 12G")
    master.add_argument("--settings", type=Path, default=reading.DEFAULT,
                        help="the settings file to write the slave into")
    master.add_argument("--port", type=int, default=rpc.DEFAULT_PORT,
                        help="the worker's port, unless the address carries one")
    master.add_argument("--reserve", type=int, default=DEFAULT_SLAVE_RESERVE,
                        help="MiB to leave on its card for its own machine")
    master.add_argument("--remove", action="store_true",
                        help="take the slave out of the settings file")
    master.set_defaults(act=_master)

    return parser.parse_args(list(argv))


def _llamacpp(given: argparse.Namespace, said: Sequence[str]) -> int:
    _on_windows("llamacpp")
    asked = upstream.NewestBuild() if given.build is None else upstream.Exactly(given.build)
    engine.install(given.settings, asked, given.check, given.force)
    return 0


def _pi(given: argparse.Namespace, said: Sequence[str]) -> int:
    client.install(given.config)
    return 0


def _on_windows(what: str) -> None:
    """llama.cpp is installed from the builds published for Windows, so a command that
    installs it refuses anywhere else before it downloads or changes anything."""
    if sys.platform != "win32":
        raise Refusal(f"install {what} installs the Windows build of llama.cpp, and this is "
                      f"{sys.platform}. What installs here: python -m cm.install pi")


def _slave(given: argparse.Namespace, said: Sequence[str]) -> int:
    """This machine made a slave: the worker at boot, and its port admitted.

    Nothing is installed here: install llamacpp is what puts a release on a machine,
    this one included, and a worker registered with no release to run would fail at
    every boot. So that is refused first, before the rights are asked for.

    Asked for the rights before anything is done, so that the elevated run is the one
    that does the whole of it. A preview asks for nothing: it changes nothing.
    """
    _on_windows("slave")
    port = Port(given.port)
    settings: Path = given.settings.resolve()
    started = task.Runs(command=Path(sys.executable),
                        arguments=task.worker_arguments(port, settings),
                        working=workspace.root())
    document = task.worker_document(started, task.Account(session.account()))

    if given.print_command:
        print("Preview only: --print-command changes nothing.")
        print(f"Task:       {task.WORKER_NAME}")
        print(f"Runs:       {started.command} {started.arguments}")
        print(f"Working in: {started.working}")
        print(f"Firewall:   {lan.worker_rule(port)}")
        print()
        print(document)
        return 0

    lending = slave.lending(settings)
    if not given.remove:
        slave.runnable(workspace.engines(), lending.build)

    changing = ("Removing this machine as a slave" if given.remove else
                "Making this machine a slave")
    match rights.asked(rights.Asking(module=__spec__.name, said=tuple(said),
                                     changing=changing)):
        case rights.Refused(why):
            raise Refusal(why)
        case session.Ran(code, wrote):
            print(wrote.rstrip())
            return code
        case rights.Held():
            pass

    if given.remove:
        _unlend(port)
        return 0

    # The task runs as this account without a logon session of its own, so it inherits
    # nothing that would create this on the way.
    files.ensure((settings.parent / lending.logs).resolve())
    autostart.register(document, task.WORKER_NAME)
    print(f"Registered the scheduled task '{task.WORKER_NAME}': the worker starts at every "
          "boot.")

    _admit(port)
    _lent(port)
    return 0


def _admit(port: Port) -> None:
    """The worker's port opened to the local subnet, replacing a rule already there."""
    if proc.run(lan.worker_shown(port)).code == 0:
        proc.asked(lan.worker_closed(port), "replace the rule")

    proc.asked(lan.worker_opened(port), "add the rule")
    print(f"Added the firewall rule: {lan.worker_rule(port)}, inbound TCP {port}, "
          f"{lan.PROFILE} profile, {lan.SUBNET} only.")


def _unlend(port: Port) -> None:
    if proc.run(("schtasks", "/Query", "/TN", task.WORKER_NAME)).code == 0:
        proc.asked(("schtasks", "/Delete", "/TN", task.WORKER_NAME, "/F"), "remove the task")
        print(f"Removed the scheduled task '{task.WORKER_NAME}'.")
    else:
        print(f"There is no scheduled task '{task.WORKER_NAME}' on this machine.")

    if proc.run(lan.worker_shown(port)).code == 0:
        proc.asked(lan.worker_closed(port), "remove the rule")
        print(f"Removed the firewall rule: {lan.worker_rule(port)}")

    print("A worker already running is left running: python -m cm.slave stop")


def _lent(port: Port) -> None:
    """What to type on the router's machine. The card's memory is read here, where the
    card is, and rounded: the router's machine cannot read it, and a round figure is all
    a reserve of two gibibytes needs."""
    largest = max(devices.cards(), key=lambda one: one.card.total)
    gibibytes = round(largest.card.total / 1024)

    print()
    print(f"Lending {largest.card.name}, {largest.card.total} MiB, on port {port}.")
    print("Start it now rather than at the next boot: python -m cm.slave start")
    print()
    print("On the machine with the router:")
    print(f"  python -m cm.install master <this machine's name or address>:{port} "
          f"{gibibytes}G")


def _master(given: argparse.Namespace, said: Sequence[str]) -> int:
    """The slave named in the settings file, or taken out of it.

    The file is read back before it is written: a settings file that stops reading is
    worse than one that does not name the slave.
    """
    settings: Path = given.settings
    text = reading.text(settings)

    if given.remove:
        files.replace(settings, config.unslaved(text))
        print(f"Took the slave out of {settings}. Place the models without it: "
              "python -m cm.calibrate")
        return 0

    if given.address is None or given.memory is None:
        raise Refusal("Name the slave and its card's memory: "
                      "python -m cm.install master <host> <memory>")

    match rpc.endpoint(given.address, Port(given.port)):
        case rpc.Unreadable(why):
            raise Refusal(f"The slave's address {why}.")
        case Endpoint() as endpoint:
            pass

    worker = Worker(endpoint=endpoint, memory=config.gibibytes(given.memory),
                    reserve=Mib(given.reserve))
    written = config.slaved(text, worker)
    reading.parsed(written)
    files.replace(settings, written)

    print(f"Wrote the slave into {settings}: {rpc.written(endpoint)}, {worker.memory} MiB, "
          f"leaving about {worker.reserve} on it.")
    if reach.reachable(endpoint):
        print("Its worker answers.")
    else:
        print("Its worker does not answer yet, and calibrate leaves its card out until it "
              "does. Start it there: python -m cm.slave start")
    print("Place the models across it: python -m cm.calibrate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
