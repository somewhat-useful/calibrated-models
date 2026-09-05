"""What a GGUF is, read out of llama-fit-params' verbose output.

Properties of the file rather than of a placement: the same answers hold for any card.

The loader prints most of them. A prediction head it does not: there is no flag to tell
the estimator one will run, so it reports the head's tensors as unused and leaves them
out of its total. Their sizes are on those same lines, which is where the head's weight
comes from, and its own cache costs what one more layer costs.

Reading is separate from running. This takes text and returns a shape, so it can be
tested without a GGUF, a card or a subprocess.
"""

import re
from dataclasses import dataclass

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
class ModelFacts:
    n_expert: int
    n_layer: Layers
    n_ctx_train: Tokens
    head: NoHead | Head


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


def parse_facts(text: str) -> ModelFacts:
    return ModelFacts(
        n_expert=_scalar(text, "n_expert"),
        n_layer=Layers(_scalar(text, "n_layer")),
        n_ctx_train=Tokens(_scalar(text, "n_ctx_train")),
        head=_head(text),
    )
