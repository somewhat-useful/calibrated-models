"""The whole preset for a machine with one card, as text.

Every rule of placement and of writing the file meets here: the search along each lever,
the head, the coarser cache that has to buy something, the mixture moving experts, the
names and the sections. A change to any of them that moves one byte of what a one-card
machine is given shows up in this file, whatever else it was meant to do.

The estimator is a law written in the test, as everywhere else in this suite, so the
numbers are arbitrary on purpose and the text below is what the rules make of them.
"""

import unittest
from pathlib import Path

from cm import place, render
from cm.config import DEFAULT_CUDA, DEFAULT_KEPT, DEFAULT_RUNTIME, Config, Model
from cm.facts import Head, ModelFacts, NoHead
from cm.machine import Card, Core, Fitted, Machine, system_memory
from cm.name import names
from cm.place import CacheType, ExpertsOnCpu
from cm.render import Placed
from cm.serving import DEFAULT_HOST, DEFAULT_IDLE, DEFAULT_PORT, DEFAULT_RESIDENT, Serving
from cm.units import Layers, Mib, Tokens
from one_card import UBATCH, chains, installed, needs
from places import somewhere

MODELS = somewhere("models")

CARD = Mib(16303)
RESERVE = Mib(1024)

MACHINE = Machine(cards=installed(Card("NVIDIA GeForce RTX 5070 Ti", CARD)),
                  ram=Mib(65407),
                  cores=tuple([Core(1, 2)] * 8 + [Core(0, 1)] * 8))

DENSE_WITH_HEAD = ModelFacts(n_expert=0, n_layer=Layers(64), n_ctx_train=Tokens(262144),
                             head=Head(weight_bytes=1_400_000_000, cache_per_token=2048))

MIXTURE = ModelFacts(n_expert=128, n_layer=Layers(40), n_ctx_train=Tokens(262144),
                     head=NoHead())

SHORT_TRAIN = ModelFacts(n_expert=0, n_layer=Layers(8), n_ctx_train=Tokens(20000),
                         head=NoHead())


def law(fixed, per_token, per_offloaded_layer=0):
    """What the estimator would answer if cost rose evenly along the lever."""

    def answer(question):
        share = question.cache.bytes_per_element / CacheType.Q8_0.bytes_per_element
        needed = fixed + per_token * question.ctx * share
        held = 0
        if isinstance(question.placement, ExpertsOnCpu):
            needed -= per_offloaded_layer * question.placement.layers
            held = per_offloaded_layer * question.placement.layers
        return needs(Mib(round(needed)), Mib(held))

    return answer


def placed(key, facts, estimator) -> Placed:
    """One model, asked about until the core stops asking, the way calibrate does it."""
    model = Model(key=key, path=MODELS / f"{key}.gguf", vendor={"temp": "1.0"},
                  allowed=place.EVERYTHING, manual=False)
    limits = place.limits_for(chains(CARD, RESERVE), UBATCH, Tokens(25000),
                              Tokens(100000))

    answers = {}
    while asking := place.next_questions(facts, model.allowed, limits, answers):
        for question in asking:
            answers[question] = estimator(question)

    chosen = place.settings(facts, model.allowed, limits, answers)
    return Placed(model, names(key, chosen), place.resident(chosen, answers))


def config() -> Config:
    return Config(model_root=MODELS,
                  cuda=DEFAULT_CUDA,
                  keep_releases=DEFAULT_KEPT,
                  preset_path=Path("llamacpp.models.ini"),
                  serving=Serving(host=DEFAULT_HOST, port=DEFAULT_PORT,
                                  resident=DEFAULT_RESIDENT, idle=DEFAULT_IDLE,
                                  logs=somewhere("llama.cpp") / "logs"),
                  reserve=RESERVE,
                  min_ctx=Tokens(25000),
                  ample_ctx=Tokens(100000),
                  cache_ram=Fitted(),
                  runtime=DEFAULT_RUNTIME,
                  shared={"split-mode": "none", "ubatch-size": 512},
                  models=(),
                  withheld=())


def written() -> str:
    every = (placed("dense", DENSE_WITH_HEAD, law(fixed=12000, per_token=0.05)),
             placed("mixture", MIXTURE,
                    law(fixed=20000, per_token=0.011, per_offloaded_layer=400)),
             placed("short", SHORT_TRAIN, law(fixed=500, per_token=0.0001)))
    memory = system_memory(Fitted(), MACHINE,
                           Mib(max(one.resident for one in every)))

    return render.preset(config(), MACHINE, memory, every)


EXPECTED = """\
{header}

version = 1

[*]
cache-ram = 50176
fit-target = 1024
split-mode = none
threads = 16
threads-batch = 16
ubatch-size = 512

[dense-115k-q4]
; VRAM REQUIRED: 15284 MiB of video memory, held from the moment this profile loads
model = {dense}
cache-type-k = q4_0
cache-type-v = q4_0
ctx-size = 115000
fit = off
gpu-layers = 99
temp = 1.0

[dense-33k-q8-mtp]
; VRAM REQUIRED: 15294 MiB of video memory, held from the moment this profile loads
model = {dense}
cache-type-k = q8_0
cache-type-v = q8_0
ctx-size = 33000
fit = off
gpu-layers = 99
spec-draft-n-max = 3
spec-draft-type-k = q8_0
spec-draft-type-v = q8_0
spec-type = draft-mtp
temp = 1.0

[dense-61k-q8]
; VRAM REQUIRED: 15290 MiB of video memory, held from the moment this profile loads
model = {dense}
cache-type-k = q8_0
cache-type-v = q8_0
ctx-size = 61000
fit = off
gpu-layers = 99
temp = 1.0

[dense-62k-q4-mtp]
; VRAM REQUIRED: 15284 MiB of video memory, held from the moment this profile loads
model = {dense}
cache-type-k = q4_0
cache-type-v = q4_0
ctx-size = 62000
fit = off
gpu-layers = 99
spec-draft-n-max = 3
spec-draft-type-k = q4_0
spec-draft-type-v = q4_0
spec-type = draft-mtp
temp = 1.0

[mixture]
; VRAM REQUIRED: 15281 MiB of video memory, held from the moment this profile loads
model = {mixture}
cache-type-k = q8_0
cache-type-v = q8_0
ctx-size = 131000
fit = off
gpu-layers = 99
n-cpu-moe = 16
temp = 1.0

[short]
; VRAM REQUIRED: 742 MiB of video memory, held from the moment this profile loads
model = {short}
cache-type-k = q8_0
cache-type-v = q8_0
ctx-size = 20000
fit = off
gpu-layers = 99
temp = 1.0
"""


class AOneCardPresetIsWhatTheRulesMakeOfIt(unittest.TestCase):
    def test_the_whole_file_is_as_written_here(self):
        self.maxDiff = None
        expected = EXPECTED.format(header=render.HEADER,
                                   dense=MODELS / "dense.gguf",
                                   mixture=MODELS / "mixture.gguf",
                                   short=MODELS / "short.gguf")

        self.assertEqual(expected, written())


if __name__ == "__main__":
    unittest.main()
