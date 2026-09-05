"""firewall: let the machines on this network reach the router, or stop letting them.

The loop is here and it decides nothing. lan.py says what the rule is and what to ask
netsh for it, and this checks what it may do, asks, and reports.

Every answer netsh gives is taken from its exit code, never from what it printed: a rule
taken for absent because a word did not match is a second rule added beside the first,
and proc.py says why a word cannot be matched at all.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import lan, proc, reading, rights, serving, session
from .config import Config, ConfigError
from .machine import UnreadableDevice
from .refusal import Refusal
from .units import Port


def main(argv: Sequence[str] | None = None) -> int:
    given = _arguments(argv if argv is not None else sys.argv[1:])
    said = argv if argv is not None else sys.argv[1:]

    try:
        match _permitted(said):
            case session.Ran(code, wrote):
                print(wrote.rstrip())
                return code
            case rights.Held():
                pass

        read = reading.read(given.settings)
        if given.remove:
            _close(read.serving.port)
        else:
            _open(read)
    except (ConfigError, Refusal, UnreadableDevice) as refusal:
        sys.stdout.flush()
        print(refusal, file=sys.stderr)
        return 1

    return 0


def _permitted(said: Sequence[str]) -> session.Ran | rights.Held:
    """The rights this run needs. What it needs them for is the firewall: what this
    machine admits is the machine's, and it says so by refusing whoever is merely
    logged in."""
    match rights.asked(rights.Asking(module=__spec__.name, said=tuple(said),
                                     changing="Changing the firewall")):
        case rights.Refused(why):
            raise Refusal(why)
        case session.Ran(_, _) | rights.Held() as answer:
            return answer


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="firewall",
        description="Admit the local subnet to the port the router listens on, so that "
                    "other machines can call it, or take that rule away. Needs an "
                    "elevated session.")
    parser.add_argument("--settings", type=Path, default=reading.DEFAULT,
                        help="the settings file naming the port")
    parser.add_argument("--remove", action="store_true",
                        help="take the rule away, leaving the router reachable from "
                             "this machine only")

    return parser.parse_args(list(argv))


def _open(read: Config) -> None:
    port = read.serving.port

    # Taken away first, so that running this again replaces the rule rather than leaving
    # a second one beside it saying something else.
    if _there(port):
        proc.asked(lan.closed(port), "replace the rule")
        print(f"Replacing the rule that was there: {lan.rule(port)}")

    proc.asked(lan.opened(port), "add the rule")

    print(f"Added the firewall rule: {lan.rule(port)}")
    print(f"Inbound TCP {port}, {lan.PROFILE} profile, {lan.SUBNET} only.")
    print(f"Clients reach it at {serving.url(read.serving)}, with this machine's name "
          "or address in place of the host.")
    print()
    print("If a client still cannot reach it, this machine's network is probably "
          "classified as Public, and the rule does not apply there. Ask Windows which "
          "it is:")
    print("  Get-NetConnectionProfile")


def _close(port: Port) -> None:
    if not _there(port):
        print(f"There is no rule called {lan.rule(port)} on this machine.")
        return

    proc.asked(lan.closed(port), "remove the rule")

    print(f"Removed the firewall rule: {lan.rule(port)}")
    print("The router is reachable from this machine only.")


def _there(port: Port) -> bool:
    """Whether the rule is already on this machine. Asking needs no elevation."""
    return proc.run(lan.shown(port)).code == 0


if __name__ == "__main__":
    sys.exit(main())
