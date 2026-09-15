"""What a GGUF is, read out of llama-fit-params' verbose output.

Properties of the file rather than of a placement: the same answers hold for any card.

The loader prints most of them. A prediction head it does not: there is no flag to tell
the estimator one will run, so it reports the head's tensors as unused and leaves them
out of its total. Their sizes are on those same lines, which is where the head's weight
comes from, and its own cache costs what one more layer costs.

Nor does it count what a head costs a model whose layers keep a recurrent state. A
draft can be rejected, a state cannot be cut back the way a cache can, and so the server
keeps snapshots of the state to roll back to. The loader says, as it allocates the state
for one sequence, which layers keep one and what it comes to, and that is read here.

Reading is separate from running. This takes text and returns a shape, so it can be
tested without a GGUF, a card or a subprocess.
"""

import re
from dataclasses import dataclass
from fractions import Fraction

from .units import Layers, Tokens


class MissingFact(Exception):
    """A number the placement is derived from was not in the output.

    Not a default and not a zero. Every window and every offload count comes out of
    these, so a plausible substitute produces a plausible placement that no one can tell
    from a real one.
    """

    def __init__(self, field: str) -> None:
        super().__init__(f"{field} not found in llama-fit-params output")
        self.field = field


class VariesByLayer(Exception):
    """The field is printed per layer, and this file needs one number from it.

    Models whose layers differ -- some attending over the whole history, some over a
    sliding window -- get a list here instead of a number. Where that model also carries
    a prediction head, which of those layers the head's own cache matches is not stated
    anywhere, and guessing would misprice the head in a way nothing downstream could
    catch.
    """

    def __init__(self, field: str) -> None:
        super().__init__(f"{field} is printed per layer, so the head cannot be weighed")
        self.field = field


@dataclass(frozen=True)
class NoHead:
    """The file carries no prediction head.

    Quantisation drops those tensors regardless of what the upstream config declares, so
    this is a property of the file in hand rather than of the model it came from.
    """


@dataclass(frozen=True)
class Head:
    """A prediction head the estimator did not count."""

    weight_bytes: int
    cache_per_token: int


@dataclass(frozen=True)
class Stateless:
    """No layer of the file keeps a recurrent state: every one caches the window."""


@dataclass(frozen=True)
class Recurrent:
    """Layers that keep a state of fixed size for every sequence, whatever the window.

    mib_per_layer is what one sequence's state costs in each of them. The loader prints
    the total in hundredths of a MiB, so it is kept exact rather than rounded twice.
    """

    layers: frozenset[int]
    mib_per_layer: Fraction


@dataclass(frozen=True)
class ModelFacts:
    n_expert: int
    n_layer: Layers
    n_ctx_train: Tokens
    head: NoHead | Head
    state: Stateless | Recurrent = Stateless()


# Whitespace before the '=' is what separates a field from the longer names it is a
# prefix of: n_layer_all and n_expert_used have a word character where this wants a
# space, so neither can answer for n_layer or n_expert.
def _scalar(text: str, field: str) -> int:
    found = re.search(rf"\b{field}\s+=\s+(\[?)\s*(\d+)", text)
    if found is None:
        raise MissingFact(field)
    if found.group(1):
        raise VariesByLayer(field)
    return int(found.group(2))


def _head(text: str) -> NoHead | Head:
    weight = sum(int(size) for size in
                 re.findall(r"unused tensor\s+\S+\s+\(size = (\d+) bytes\)", text))
    if weight == 0:
        return NoHead()

    # One more layer, so its cache costs what one layer's does: keys and values together.
    return Head(weight_bytes=weight,
                cache_per_token=_scalar(text, "n_embd_k_gqa")
                + _scalar(text, "n_embd_v_gqa"))


# A layer given a recurrent state, and what the states of one sequence come to on a device.
# A layer that keeps none is printed as skipped, and is not matched.
_STATEFUL = re.compile(r"llama_memory_recurrent, layer\s+(\d+): dev = ")
_STATE_BUFFER = re.compile(
    r"llama_memory_recurrent:\s+\S+ RS buffer size =\s+(\d+(?:\.\d+)?) MiB")


def _state(text: str) -> Stateless | Recurrent:
    """Which layers keep a recurrent state, and what one sequence's costs each of them.

    A file with no recurrent layers prints neither kind of line. One kind without the
    other is output this was not written for, and a guess would misprice every head.
    """
    layers = frozenset(int(found) for found in _STATEFUL.findall(text))
    sizes = [Fraction(found) for found in _STATE_BUFFER.findall(text)]

    if not layers and not sizes:
        return Stateless()
    if not layers:
        raise MissingFact("llama_memory_recurrent layer")
    if not sizes:
        raise MissingFact("RS buffer size")

    return Recurrent(layers=layers, mib_per_layer=sum(sizes, Fraction(0)) / len(layers))


def parse_facts(text: str) -> ModelFacts:
    return ModelFacts(
        n_expert=_scalar(text, "n_expert"),
        n_layer=Layers(_scalar(text, "n_layer")),
        n_ctx_train=Tokens(_scalar(text, "n_ctx_train")),
        head=_head(text),
        state=_state(text),
    )
