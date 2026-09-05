"""Running the programs of a llama.cpp release, and the ones Windows ships.

Both streams come back because the two things the estimator is asked for are printed on
different ones: the requirement on stdout, the architecture in the verbose log on
stderr. The exit code comes back with them, because a program that answers nothing --
schtasks asked to end a task that is not running -- says so only that way.

What a program said is passed through for a person to read and never matched against.
The scheduler and the firewall both answer in the language Windows was installed in, so
a decision taken on an English word is one that goes wrong on a machine installed in
another; the exit code is the same number everywhere.
"""

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .advise import Pid
from .refusal import Refusal

# Windows: no console of its own, and no share in this one's Ctrl-C. A router started
# from a session that then ends is a router that carries on serving, which is the point
# of starting it at all.
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200


@dataclass(frozen=True)
class Output:
    """What a run of a program printed, and what it exited with."""

    out: str
    err: str
    code: int


def run(argv: Sequence[str]) -> Output:
    done = subprocess.run(list(argv), capture_output=True, text=True)
    return Output(out=done.stdout, err=done.stderr, code=done.returncode)


def asked(argv: Sequence[str], what: str) -> None:
    """One thing asked of a program Windows ships, with what it said passed through.

    The program is named in the refusal out of the command line itself, so what a person
    reads is the name of the thing that would not do it rather than the name of the
    command that asked.
    """
    done = run(argv)

    for line in (done.out + done.err).splitlines():
        if line.strip():
            print(f"  {line.strip()}")

    if done.code != 0:
        raise Refusal(f"{argv[0]} could not {what} (exit {done.code}). What it said is "
                      "above.")


def attached(argv: Sequence[str], working: Path) -> int:
    """Run a program in this console and wait for it, giving back its exit code."""
    return subprocess.call(list(argv), cwd=str(working))


def detached(argv: Sequence[str], working: Path, out: Path, err: Path) -> Pid:
    """Start a program that outlives this one, with its two streams into files.

    Appended to rather than replaced: the streams of the run that failed yesterday are
    what the person starting it again this morning wants to read.
    """
    with out.open("ab") as stdout, err.open("ab") as stderr:
        started = subprocess.Popen(list(argv), cwd=str(working),
                                   stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr,
                                   creationflags=(_DETACHED_PROCESS
                                                  | _CREATE_NEW_PROCESS_GROUP))

    return Pid(started.pid)
