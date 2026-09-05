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

from .policy import Room
from .served import Router, Served

# The shape pi fixes for a provider, and what llama.cpp's OpenAI endpoint supports. The
# same wherever it runs, so it is written down rather than asked.
COMPAT = {"supportsDeveloperRole": False,
          "supportsReasoningEffort": False,
          "supportsStore": False,
          "supportsStrictMode": False,
          "maxTokensField": "max_tokens"}

API = "openai-completions"

# Ignored by llama.cpp, and pi will not call a provider that carries none.
KEY = "llama.cpp"


class UnknownShape(Exception):
    """A configuration of pi's this does not know how to edit, and what it expected."""


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


def pointed(document: Mapping[str, object], router: Router, fallback: str,
            models: Sequence[Served]) -> Written:
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
    provider["models"] = [_model(one) for one in models]

    return Written(written, tuple(changes))


def compacted(document: Mapping[str, object], room: Room) -> Written:
    """pi's settings.json, with its two global numbers behind the policy's.

    A value already there is left alone while it is behind the policy: what a person set
    for their own reasons is theirs until it stops working. Only one that would take the
    extension out of the decision is corrected, and to a round number under the bound
    rather than to the bound itself -- these are also what pi falls back on when
    summarising fails, so there is no reason to sit one token off the line.
    """
    written = copy.deepcopy(dict(document))

    settings = written.get("compaction")
    if not isinstance(settings, Mapping):
        written["compaction"] = {"enabled": True,
                                 "reserveTokens": room.reserve.take,
                                 "keepRecentTokens": room.keep.take}
        return Written(written, (f"compaction set to reserve {room.reserve.take}, "
                                 f"keeping {room.keep.take} of the recent conversation",))

    compaction = dict(settings)
    written["compaction"] = compaction
    changes = []

    held = compaction.get("reserveTokens")
    if not _count(held):
        compaction["reserveTokens"] = room.reserve.take
        changes.append(f"reserveTokens was not set -> {room.reserve.take}")
    elif held > room.reserve.most:
        compaction["reserveTokens"] = room.reserve.take
        changes.append(f"reserveTokens {held} -> {room.reserve.take}: from "
                       f"{room.reserve.most + 1} up the policy stands aside and pi "
                       "compacts on its own")

    recent = compaction.get("keepRecentTokens")
    if not _count(recent):
        compaction["keepRecentTokens"] = room.keep.take
        changes.append(f"keepRecentTokens was not set -> {room.keep.take}")
    elif recent > room.keep.most:
        compaction["keepRecentTokens"] = room.keep.take
        changes.append(f"keepRecentTokens {recent} -> {room.keep.take}: above "
                       f"{room.keep.most} pi answers that there is nothing worth "
                       "compacting and the policy is never asked where to cut")

    if not isinstance(compaction.get("enabled"), bool):
        compaction["enabled"] = True
        changes.append("compaction was not enabled -> true")

    return Written(written, tuple(changes))


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


def _model(served: Served) -> dict[str, object]:
    """One model as pi lists it."""
    return {"id": served.id,
            "name": served.name,
            "contextWindow": served.window,
            "maxTokens": served.cap,
            "input": list(served.modalities),
            "reasoning": served.reasoning}
