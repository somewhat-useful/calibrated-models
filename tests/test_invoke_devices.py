"""Invariants of what the estimator is asked about a configuration across several devices.

A card of this machine is asked about by the name llama.cpp gives it, a slave's by the
worker it is reached through; the layers are counted per device in the order they run;
and one card is asked about exactly as it always was.
"""

import unittest

from cm.config import DEFAULT_RUNTIME, Runtime
from cm.invoke import argv
from cm.machine import CudaIndex
from cm.nonempty import NonEmpty
from cm.place import (NO_TENSOR, CacheType, Endpoint, Layout, Local, Pipeline, Question,
                      Remote, WholeCard)
from cm.units import Halvings, Layers, Mib, Port, Tokens
from places import somewhere

BINARY = somewhere("llama.cpp", "release", "llama-fit-params.exe")
MODEL = somewhere("models", "model.gguf")

FAST = Local(CudaIndex(0), Mib(16303))
SLOW = Local(CudaIndex(1), Mib(8192))
SLAVE = Remote(Endpoint("worker", Port(50052)), Mib(12288))


def asked(devices, layers, halvings=0, pipeline=Pipeline.ON, runtime=DEFAULT_RUNTIME):
    layout = Layout(devices=NonEmpty(*devices), layers=NonEmpty(*map(Layers, layers)),
                    halvings=Halvings(halvings), pipeline=pipeline)
    return argv(BINARY, MODEL, Question(Tokens(45000), CacheType.Q8_0, WholeCard(), layout),
                runtime)


def after(flag, line):
    """What was passed with a flag, once."""
    given = [line[index + 1] for index, word in enumerate(line) if word == flag]
    if len(given) != 1:
        raise AssertionError(f"{flag} appears {len(given)} times in {line}")
    return given[0]


class SeveralCardsAreNamedInTheOrderTheLayersRunThroughThem(unittest.TestCase):
    def test_the_devices_and_their_counts_go_in_the_same_order(self):
        line = asked((SLOW, FAST), (25, 41))

        self.assertEqual("layer", after("--split-mode", line))
        self.assertEqual("CUDA1,CUDA0", after("--device", line))
        self.assertEqual("25,41", after("--tensor-split", line))

    def test_the_order_is_the_layouts_and_not_the_cards_numbers(self):
        line = asked((FAST, SLOW), (41, 25))

        self.assertEqual("CUDA0,CUDA1", after("--device", line))
        self.assertEqual("41,25", after("--tensor-split", line))


class OneCardIsAskedAboutAsItAlwaysWas(unittest.TestCase):
    def test_no_splitting_and_nothing_else(self):
        line = asked((FAST,), (66,), pipeline=Pipeline.OFF)

        self.assertEqual("none", after("--split-mode", line))
        for flag in ("--device", "--tensor-split", "--rpc", "-ot"):
            with self.subTest(flag=flag):
                self.assertNotIn(flag, line)


class CardsKeptApartAreKeptApartByAnOverrideThatMovesNothing(unittest.TestCase):
    def test_apart_overrides_a_tensor_nothing_is_called_onto_the_first_card(self):
        line = asked((SLOW, FAST), (25, 41), pipeline=Pipeline.OFF)

        self.assertEqual(f"{NO_TENSOR}=CUDA1", after("-ot", line))

    def test_together_overrides_nothing(self):
        self.assertNotIn("-ot", asked((SLOW, FAST), (25, 41), pipeline=Pipeline.ON))


class ASlavesCardIsReachedThroughItsWorkerAndRunsFirst(unittest.TestCase):
    def test_the_worker_is_named_and_its_card_goes_first(self):
        line = asked((SLAVE, SLOW, FAST), (24, 15, 27), pipeline=Pipeline.OFF)

        self.assertEqual("worker:50052", after("--rpc", line))
        self.assertEqual("RPC0,CUDA1,CUDA0", after("--device", line))
        self.assertEqual("24,15,27", after("--tensor-split", line))

    def test_nothing_is_overridden_where_llama_cpp_keeps_the_cards_apart_itself(self):
        self.assertNotIn("-ot", asked((SLAVE, FAST), (31, 35), pipeline=Pipeline.OFF))

    def test_a_chain_of_the_machines_own_cards_names_no_worker(self):
        self.assertNotIn("--rpc", asked((SLOW, FAST), (25, 41)))


class TheMicroBatchIsHalvedFromTheOneTheSettingsFileRunsWith(unittest.TestCase):
    def test_each_halving_halves_it(self):
        for ubatch, halvings, expected in ((512, 0, "512"), (512, 1, "256"),
                                           (512, 2, "128"), (1024, 1, "512")):
            runtime = Runtime(batch=2048, ubatch=ubatch, parallel=1, flash_attn="on")
            with self.subTest(ubatch=ubatch, halvings=halvings):
                self.assertEqual(expected, after("-ub", asked((SLOW, FAST), (25, 41),
                                                              halvings=halvings,
                                                              runtime=runtime)))


if __name__ == "__main__":
    unittest.main()
