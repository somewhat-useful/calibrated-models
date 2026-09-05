"""The command lines the estimator is asked with.

Building them is arithmetic on a question, so it is here and not in the module that runs
processes. What matters is that the estimate is asked for the same run the router will
later make: batching and flash attention move the compute buffers by hundreds of
megabytes, so a placement computed at one batch size and served at another was computed
for a machine that does not exist.
"""

from collections.abc import Sequence
from pathlib import Path

from .config import Runtime
from .place import ExpertsOnCpu, Question

# Every layer on the card, as llama.cpp spells it.
ALL_LAYERS = "99"

# What the estimator is asked for a header reading: the shortest window it will accept,
# since nothing in the answer is used -- only the architecture it prints on the way.
HEADER_CTX = "4096"


def facts_argv(binary: Path, model: Path) -> tuple[str, ...]:
    """Ask for the model's architecture. Verbose, because that is where it is printed."""
    return (str(binary), "-m", str(model),
            "-c", HEADER_CTX, "-ngl", ALL_LAYERS,
            "--split-mode", "none", "--fit", "off", "-v")


def argv(binary: Path, model: Path, question: Question,
         runtime: Runtime) -> tuple[str, ...]:
    """Ask what one configuration would need.

    `--fit off` because the placement is the question: left on, the loader would answer
    about a configuration of its own choosing instead of the one being asked about.
    """
    return (str(binary), "-m", str(model),
            "-c", str(question.ctx),
            "-b", str(runtime.batch), "-ub", str(runtime.ubatch),
            "-ctk", question.cache.value, "-ctv", question.cache.value,
            "-fa", runtime.flash_attn,
            "-np", str(runtime.parallel),
            "--split-mode", "none",
            "--fit", "off", "--fit-print", "on",
            *_placement(question),
            )


def _placement(question: Question) -> Sequence[str]:
    """Where the layers go. Every layer on the card either way; a mixture then moves
    the experts of some of them back into system memory."""
    if isinstance(question.placement, ExpertsOnCpu):
        return ("-ngl", ALL_LAYERS, "-ncmoe", str(question.placement.layers))
    return ("-ngl", ALL_LAYERS)
