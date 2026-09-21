"""The order the commands are run in, so that nobody has to open the README for it.

No state between them, which is what makes the order worth writing down: nothing here
refuses because something else has not run yet, so the only thing saying what comes next
is this.

The path is the ordinary one, start to finish on a machine that has nothing yet. Some
commands sit off it -- one asks the card what it can load right now, one stops the
router, two start and stop a slave's worker by hand -- and they are listed apart
rather than numbered into a sequence they are not part of.
"""

from collections.abc import Sequence
from dataclasses import dataclass

# What a person types, and what every module here is run as. Written out rather than
# derived from __name__: this is the text to copy, and the package it names is the one
# the reader is standing in.
RUN = "python -m cm"


@dataclass(frozen=True)
class Step:
    """One command in the path: what to run, what it leaves behind, and whether it
    takes --settings.

    The last is written against the command rather than said once at the end, because
    said once at the end it is a sentence nobody has to keep true. Against the command
    it is a claim about that command, and a test asks the command itself.
    """

    command: str
    does: str
    settings: bool


@dataclass(frozen=True)
class Stage:
    """A group of steps that answer one question, and the question."""

    about: str
    steps: tuple[Step, ...]


PATH = (
    Stage("1. The engine, and a settings file", (
        Step("install llamacpp", "the newest llama.cpp for this card, and a settings "
                                 "file", settings=True),
    )),
    Stage("2. The models", (
        Step("scan", "an entry per GGUF, with what its publisher recommends",
             settings=True),
        Step("models", "what is here, what is missing, and where it came from",
             settings=True),
        Step("calibrate", "the window and the offload for each, on this card",
             settings=True),
    )),
    Stage("3. Serve", (
        Step("router start", "run the server on what calibrate worked out",
             settings=True),
    )),
    Stage("4. Keep serving, and let the network in", (
        Step("autostart", "start it at every boot         (asks for rights)",
             settings=True),
        Step("firewall", "admit the local subnet to its port (asks for rights)",
             settings=True),
    )),
    Stage("5. On whatever machine calls it", (
        Step("install pi", "the pi agent and the context-policy extension",
             settings=False),
        Step("pi <host>", "point pi at the router and list what it serves",
             settings=False),
    )),
    Stage("6. Another machine lending its card", (
        Step("install llamacpp", "on it: the build the router runs, with --build",
             settings=True),
        Step("install slave", "on it: its worker at every boot (asks for rights)",
             settings=True),
        Step("install master", "here: name it in the settings, then calibrate again",
             settings=True),
    )),
)

BESIDE = (
    Step("vram", "what the card can load now, and what to close", settings=True),
    Step("router stop", "stop the server holding the router's port", settings=True),
    Step("slave start", "on a slave: start its worker by hand", settings=True),
    Step("slave stop", "on a slave: stop its worker", settings=True),
)


def path() -> tuple[str, ...]:
    """The whole thing, as lines to print."""
    widest = _widest(_every())

    lines = ["The path, start to finish on a machine that has nothing yet:"]
    for stage in PATH:
        lines += ["", stage.about, *_written(stage.steps, widest)]

    return tuple([*lines,
                  "", "Whenever you want them, not part of the path:",
                  *_written(BESIDE, widest),
                  "", flags()])


def _every() -> tuple[Step, ...]:
    """Every step there is, on the path and beside it."""
    return (*(one for stage in PATH for one in stage.steps), *BESIDE)


def _widest(steps: Sequence[Step]) -> int:
    """The column the second half of every line starts in. Nought where there are no
    steps to line up, rather than a refusal: emptying either list is a way of editing
    them, and it is not this function's place to have an opinion about it."""
    return max((len(one.command) for one in steps), default=0)


def flags() -> str:
    """What they all take, said once at the end rather than against each of them, where
    it would be the same sentence written out against every one."""
    without = tuple(one.command for one in _every() if not one.settings)

    if not without:
        return "Every one of them takes --help and --settings."
    if len(without) == len(_every()):
        return "Every one of them takes --help."

    return ("Every one of them takes --help, and all but "
            f"{_listed(without)} take --settings.")


def _listed(commands: Sequence[str]) -> str:
    """Them, written the way a sentence takes a list."""
    return (commands[0] if len(commands) == 1 else
            " and ".join((", ".join(commands[:-1]), commands[-1])))


def _written(steps: Sequence[Step], widest: int) -> list[str]:
    return [f"  {RUN}.{one.command:<{widest}}  {one.does}" for one in steps]


def commands() -> frozenset[str]:
    """Every module the path names, which is what the package has to carry."""
    return frozenset(one.command.split()[0] for one in _every())
