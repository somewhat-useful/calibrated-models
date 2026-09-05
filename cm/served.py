"""What the router serves, read out of what it answers.

A router reports, for every model it offers, the command line it will run that model
with. That line is the whole truth about the model -- the window it will hold, the file
it will load, how exactly it holds the conversation, whether a prediction head drafts
ahead -- so everything a client needs is read from it. Nothing is copied from the
machine with the card, and the two cannot come to hold different ideas of what is
served.

Parsing inward is fallible: a model the router lists but a client cannot be pointed at
comes back as Unusable with the reason, rather than as a model with a made-up window.
"""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .units import Tokens

# How llama.cpp spells "every layer on the card". Fewer than this and part of the model
# is answered from system memory, which is worth saying out loud in a model's name.
WHOLE_CARD = 99

# A window is chosen and written a thousand tokens at a time, so it is named that way:
# 131072 reads as 131k. Not kibibytes -- 28000 has to read back as 28k, not 27k.
THOUSAND = 1000

# What one answer may take of the window it shares with the conversation, and the bounds
# that share is held between. The floor is not free to move: the context-policy
# extension sizes its own reserve from this cap, and a reserve under 8192 is one pi
# would cut before the policy ever ran.
SHARE = 3
CAP_FLOOR = Tokens(8192)
CAP_CEILING = Tokens(32768)

# What the quantisation looks like at the tail of a file name: Q4_K_M, IQ4_XS, UD-IQ4_XS,
# BF16. What precedes it names the model.
_QUANT = re.compile(r"^(?P<base>.+?)-(?P<quant>(UD-)?(IQ|Q|BF|F)\d+(_[A-Za-z0-9]+)*)$")

# An attention cache named for its width: the block layout comes with the width, and it
# is the width a person is choosing between.
_CACHE = re.compile(r"^q(?P<bits>\d+)_")

# What a repository directory adds to a file name, once the suffix Hugging Face gives
# every one of them is off.
_REPOSITORY = re.compile(r"-GGUF$", re.IGNORECASE)


@dataclass(frozen=True)
class Router:
    """Where the router is, as a client on another machine addresses it."""

    host: str
    port: int

    @property
    def base_url(self) -> str:
        """What an OpenAI-compatible client is pointed at."""
        return f"http://{self.host}:{self.port}/v1"

    @property
    def models_url(self) -> str:
        """Where it lists what it serves."""
        return f"{self.base_url}/models"


@dataclass(frozen=True)
class Served:
    """One model the router serves, as a client has to describe it."""

    id: str
    name: str
    window: Tokens
    cap: Tokens
    modalities: tuple[str, ...]
    reasoning: bool


@dataclass(frozen=True)
class Unusable:
    """A model the router lists that a client cannot be pointed at, and why."""

    id: str
    why: str


Offered = Served | Unusable


def read(entry: Mapping[str, object]) -> Offered:
    """One entry of the router's model list."""
    served = entry.get("id")
    if not isinstance(served, str) or not served:
        return Unusable("", "an entry with no id")

    argv = _argv(entry)
    if not argv:
        return Unusable(served, "reports no command line, so nothing can be read of it")

    window = flag(argv, "--ctx-size")
    if not window.isdigit() or int(window) <= 0:
        return Unusable(served, "reports no window")

    ctx = Tokens(int(window))

    return Served(id=served,
                  name=display(argv, ctx),
                  window=ctx,
                  cap=reply_cap(ctx),
                  modalities=_modalities(entry),
                  # A model that does not reason simply never fills reasoning_content,
                  # so the flag is the only thing worth reading.
                  reasoning=flag(argv, "--reasoning") != "off")


def flag(argv: Sequence[str], name: str) -> str:
    """The value following a flag, or "" where it is absent or takes none.

    A flag followed by another flag was given no value: llama.cpp spells switches that
    way, and reading the next flag as a value would be a value nobody wrote.
    """
    for at, written in enumerate(argv):
        if written != name:
            continue
        if at + 1 >= len(argv) or argv[at + 1].startswith("--"):
            return ""
        return argv[at + 1]

    return ""


def display(argv: Sequence[str], ctx: Tokens) -> str:
    """The name a model picker shows, built from what the model runs with.

    Everything in it comes from a flag: the family and the quantisation from the weights
    path, then whatever tells one profile of the same weights from another -- the
    attention cache, a prediction head, layers answered from system memory -- and the
    window last, because that is what a person chooses by.

    n-cpu-moe is deliberately not named: every mixture-of-experts profile places experts
    on the CPU, so saying so distinguishes nothing.
    """
    weights = flag(argv, "--model")
    if not weights:
        return ""

    parts = [_weights(weights)]

    cache = _CACHE.match(flag(argv, "--cache-type-k"))
    if cache is not None:
        parts.append(f"q{cache['bits']} cache")

    if flag(argv, "--spec-type") == "draft-mtp":
        parts.append("MTP")

    layers = flag(argv, "--n-gpu-layers")
    if layers.isdigit() and 0 < int(layers) < WHOLE_CARD:
        parts.append("part on CPU")

    return f"{', '.join(parts)} ({ctx // THOUSAND}k)"


def reply_cap(window: Tokens) -> Tokens:
    """How many tokens one answer may take of the window it shares.

    A third of the window, rounded to a power of two and held between the bounds above.
    The conversation needs the rest, so the cap cannot be the whole window; and past the
    ceiling it stops being a cap on anything anyone would wait for.

    A window shorter than the floor is its own cap: an answer cannot be longer than the
    window it is written into, whatever the extension would prefer.
    """
    third = max(1, window // SHARE)
    nearest = 2 ** round(math.log2(third))

    return Tokens(min(window, max(CAP_FLOOR, min(CAP_CEILING, nearest))))


def _weights(path: str) -> str:
    """What the file and the directory holding it say the model is.

    A repository directory often carries a variant the file name drops -- A3B, Instruct,
    Thinking -- and that variant is worth showing.
    """
    parts = path.replace("\\", "/").split("/")
    stem = parts[-1].rsplit(".", 1)[0]
    repository = parts[-2] if len(parts) > 1 else ""

    quantised = _QUANT.match(stem)
    base = quantised["base"] if quantised else stem
    quant = quantised["quant"] if quantised else ""

    fuller = _REPOSITORY.sub("", repository)
    if fuller.startswith(base) and len(fuller) > len(base):
        base = fuller

    named = base.replace("-", " ")
    return f"{named} {quant}" if quant else named


def _argv(entry: Mapping[str, object]) -> tuple[str, ...]:
    status = entry.get("status")
    if not isinstance(status, Mapping):
        return ()

    argv = status.get("args")
    if not isinstance(argv, list):
        return ()

    return tuple(one for one in argv if isinstance(one, str))


def _modalities(entry: Mapping[str, object]) -> tuple[str, ...]:
    """What the model accepts, as the router reports it. Text unless it says otherwise."""
    architecture = entry.get("architecture")
    if not isinstance(architecture, Mapping):
        return ("text",)

    reported = architecture.get("input_modalities")
    if not isinstance(reported, list):
        return ("text",)

    named = tuple(one for one in reported if isinstance(one, str) and one)
    return named or ("text",)
