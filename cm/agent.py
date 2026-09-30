"""pi's own configuration, as it should read once this router is what it talks to.

Two files, both pi's. models.json holds the providers it can call and the models each
one serves; settings.json holds, among much else, where it compacts a conversation.
Neither is ours: everything not about this router is carried over untouched, because
either file may hold entries this program knows nothing about.

Documents in, documents out. Nothing is opened here, and nothing decides when to write:
that is the caller's, which is also where a backup is taken.
"""

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .policy import Bound, Room
from .recommended import EFFORTS, Thinking, Told, Untold
from .served import Router, Served

# Whether a client may choose the reasoning effort for itself. The one flag of the shape
# below a single model may answer differently -- pi reads what a model says over what its
# provider said -- and false for the provider, so that a model whose chat template nobody
# has read is left on the effort the router was started with.
REASONING_EFFORT = "supportsReasoningEffort"

# The shape pi fixes for a provider, and what llama.cpp's OpenAI endpoint supports. The
# same wherever it runs, so it is written down rather than asked.
COMPAT = {"supportsDeveloperRole": False,
          REASONING_EFFORT: False,
          "supportsStore": False,
          "supportsStrictMode": False,
          "maxTokensField": "max_tokens"}

# What pi calls asking for no thinking at all, and what a request carries to ask for it.
# llama.cpp answers this one itself, switching thinking off before the model's template is
# asked anything, so it is the level no row has to name and every model can be offered.
OFF, NONE = "off", "none"

# What a model accepts that says it is not something to hold a conversation with. A
# recogniser is served by the router like any other model and answers on the same endpoint,
# but a coding agent has nothing to send it and no way to read what it sends back: its
# whole input is a recording, and the reply is a transcript rather than a turn. So it is
# left out of pi's list, where it would otherwise sit among the models a person picks from.
#
# Pictures are not this. A model that reads an image reads a conversation too, and pi
# passes the modalities through so that it can send one.
AUDIO = "audio"

# pi's two global compaction numbers: how much of the window it holds back for the reply,
# and how much of the recent conversation it keeps word for word.
RESERVE, KEEP = "reserveTokens", "keepRecentTokens"

API = "openai-completions"

# Ignored by llama.cpp, and pi will not call a provider that carries none.
KEY = "llama.cpp"


class UnknownShape(Exception):
    """A configuration of pi's this does not know how to edit, and what it expected."""


@dataclass(frozen=True)
class Offering:
    """One model as a client is to be told about it: what the router serves, and what
    this repository says a client may ask that model to think at."""

    served: Served
    thinking: Told


@dataclass(frozen=True)
class Written:
    """A document as it should be after this run, and what changed getting there."""

    document: Mapping[str, object]
    changes: tuple[str, ...]


def empty() -> dict[str, object]:
    """The models.json of a pi that has never been started.

    pi writes it on first run, and this is the shape it documents: providers, with none
    in them yet. `pointed` refuses a document without that object rather than guessing
    at it, so this is what a machine where pi is installed but has not run needs before
    anything can be written into it.
    """
    return {"providers": {}}


def chats(served: Served) -> bool:
    """Whether pi has any use for this model. See AUDIO for the one that it has not."""
    return AUDIO not in served.modalities


def chatting(models: Sequence[Served]) -> tuple[Served, ...]:
    """Only the models pi has a use for, out of everything the router serves.

    Here rather than at each place that needs it, because two places do: the list pi is
    given, and the compaction numbers it has to fit behind. A recogniser holds a window
    as short as its longest recording, so leaving it in the second would compact every
    conversation down to a window nothing uses.
    """
    return tuple(one for one in models if chats(one))


def pointed(document: Mapping[str, object], router: Router, fallback: str,
            offered: Sequence[Offering]) -> Written:
    """pi's models.json, with this router's provider rewritten and no other touched.

    Which provider is this router: the one already pointing at it, else the one named,
    else a new one under that name. Identified by where it points rather than by what it
    is called -- a provider renamed in pi is still this router -- and falling back to the
    name is what lets the address change without leaving a second, stale provider behind.
    """
    providers = document.get("providers")
    if not isinstance(providers, Mapping):
        raise UnknownShape("pi's models.json has no providers object; "
                           "refusing to guess at its shape")

    written = copy.deepcopy(dict(document))
    into = dict(providers)
    written["providers"] = into

    name = _which(into, router, fallback)
    changes = []

    if name not in into:
        into[name] = {"api": API, "apiKey": KEY, "baseUrl": router.base_url,
                      "compat": dict(COMPAT), "models": []}
        changes.append(f"created the provider {name} -> {router.base_url}")

    provider = dict(into[name])
    into[name] = provider

    was = provider.get("baseUrl")
    if was != router.base_url:
        changes.append(f"{name} now points at {router.base_url}, was {was}")

    # The address just given wins: it is the reason this was run. Everything else about
    # an existing provider is left alone -- compat flags someone set by hand are theirs.
    provider["baseUrl"] = router.base_url
    provider["models"] = [_model(one) for one in offered]

    return Written(written, tuple(changes))


def compacted(document: Mapping[str, object], room: Room) -> Written:
    """pi's settings.json, with its two global numbers where the policy leaves room for.

    A value inside what the policy leaves room for is left alone: what a person set for
    their own reasons is theirs while it works.

    Outside it, this writes in both directions. A run that only ever lowers is a ratchet:
    the fleet a number was lowered for changes -- a model with a short window is added,
    served for a while and taken out -- and pi is left compacting for a router that is no
    longer there, with nothing to say so and nobody to notice. Raising it by hand is not
    the answer either, since the next run lowers it again.

    Either way the value written is a round one under the bound rather than the bound
    itself: these are also what pi falls back on when summarising fails, so there is no
    reason to sit one token off the line.
    """
    written = copy.deepcopy(dict(document))

    settings = written.get("compaction")
    if not isinstance(settings, Mapping):
        written["compaction"] = {"enabled": True,
                                 RESERVE: room.reserve.take,
                                 KEEP: room.keep.take}
        return Written(written, (f"compaction set to reserve {room.reserve.take}, "
                                 f"keeping {room.keep.take} of the recent conversation",))

    compaction = dict(settings)
    written["compaction"] = compaction

    changes = [
        _put(compaction, RESERVE, room.reserve,
             over=f"from {room.reserve.most + 1} up the policy stands aside and pi "
                  "compacts on its own",
             under=f"the policy leaves room up to {room.reserve.most}, and this is also "
                   "what pi holds back for a reply when summarising fails"),
        _put(compaction, KEEP, room.keep,
             over=f"above {room.keep.most} pi answers that there is nothing worth "
                  "compacting and the policy is never asked where to cut",
             under=f"the policy leaves room up to {room.keep.most}, and keeping less of "
                   "the recent conversation than that is context given up for nothing"),
    ]

    if not isinstance(compaction.get("enabled"), bool):
        compaction["enabled"] = True
        changes.append("compaction was not enabled -> true")

    return Written(written, tuple(one for one in changes if one))


def _put(compaction: dict[str, object], key: str, bound: Bound,
         over: str, under: str) -> str:
    """One of pi's numbers, put where the policy leaves room for it, and the line saying so.

    Empty where it is already inside that room: a run that changed nothing says nothing.
    """
    was = compaction.get(key)
    if not _count(was):
        compaction[key] = bound.take
        return f"{key} was not set -> {bound.take}"

    if bound.take <= was <= bound.most:
        return ""

    compaction[key] = bound.take
    return f"{key} {was} -> {bound.take}: {over if was > bound.most else under}"


def _which(providers: Mapping[str, object], router: Router, fallback: str) -> str:
    """The provider already pointing at this router, or the name to write instead."""
    for name, provider in providers.items():
        if not isinstance(provider, Mapping):
            continue
        if _points_here(provider.get("baseUrl"), router):
            return name

    return fallback


def _points_here(url: object, router: Router) -> bool:
    """Whether a provider's address is this router's, whatever path it carries.

    The port has to end where the address says it ends: :18081 and :180810 are two
    machines, and a prefix test alone would take the second for the first.
    """
    if not isinstance(url, str):
        return False

    at = f"//{router.host}:{router.port}"
    if at not in url:
        return False

    rest = url.split(at, 1)[1]
    return rest == "" or rest.startswith("/")


def _count(value: object) -> bool:
    """Whether a number pi wrote is one this can compare against: a positive whole."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _model(offering: Offering) -> dict[str, object]:
    """One model as pi lists it, with the efforts it may be asked for where they are
    known. A model whose template nobody has read is listed without them, and pi then
    sends no effort at all: what the router was started with stands."""
    served = offering.served
    listed: dict[str, object] = {"id": served.id,
                                 "name": served.name,
                                 "contextWindow": served.window,
                                 "maxTokens": served.cap,
                                 "input": list(served.modalities),
                                 "reasoning": served.reasoning}

    match offering.thinking:
        case Untold():
            return listed
        case Thinking(efforts):
            return {**listed,
                    "thinkingLevelMap": _levels(efforts),
                    "compat": {REASONING_EFFORT: True}}


def _levels(efforts: Sequence[str]) -> dict[str, object]:
    """What each level a person can pick in pi carries in a request.

    Every level pi knows of is named here, because one left out of the map is one pi
    offers and passes through as it stands: null is how it is told not to offer a level at
    all. So a model is offered the efforts its own template takes and nothing else, and
    off, which llama.cpp answers rather than the template.
    """
    taken = frozenset(efforts)

    return {OFF: NONE, **{one: (one if one in taken else None) for one in EFFORTS}}
