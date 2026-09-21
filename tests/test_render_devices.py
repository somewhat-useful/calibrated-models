"""Invariants of a section that runs a model across several devices.

It says everything the router needs to run the placement the way it was computed: which
devices in which order, how many layers on each, the worker a slave's card is reached
through, and the micro-batch where it is not the shared one. A section about one card
adds none of it, which the whole one-card preset in test_one_card holds down.
"""

import unittest

from cm.machine import CudaIndex
from cm.nonempty import NonEmpty
from cm.place import (NO_TENSOR, Among, CacheType, Endpoint, Layout, Local, Pipeline,
                      Remote, Settings, WholeCard)
from cm.render import REQUIRED, preset
from cm.units import Halvings, Layers, Mib, Port, Tokens
from test_render import MACHINE, MEMORY, config, placed, read, stated

FAST = Local(CudaIndex(0), Mib(16303))
SLOW = Local(CudaIndex(1), Mib(8192))
SLAVE = Remote(Endpoint("worker", Port(50052)), Mib(12288))


def across(devices, layers, spare, halvings=0, pipeline=Pipeline.ON, ctx=150000,
           among=Among.SEVERAL):
    return Settings(ctx=Tokens(ctx), cache=CacheType.Q8_0, head=False,
                    placement=WholeCard(),
                    spare=NonEmpty(*map(Mib, spare)),
                    layout=Layout(devices=NonEmpty(*devices),
                                  layers=NonEmpty(*map(Layers, layers)),
                                  halvings=Halvings(halvings), pipeline=pipeline,
                                  among=among))


def section(settings, named="model"):
    _, parsed = read(preset(config(), MACHINE, MEMORY, (placed("model", settings),)))
    return parsed[named]


class TheSectionSaysWhereTheLayersGo(unittest.TestCase):
    def test_devices_counts_and_splitting_are_written(self):
        written = section(across((SLOW, FAST), (25, 41), (1024, 2048)), named="model-2gpu")

        self.assertEqual("CUDA1,CUDA0", written["device"])
        self.assertEqual("layer", written["split-mode"])
        self.assertEqual("25,41", written["tensor-split"])

    def test_the_cards_running_together_override_nothing(self):
        written = section(across((SLOW, FAST), (25, 41), (1024, 2048)), named="model-2gpu")

        self.assertNotIn("override-tensor", written)
        self.assertNotIn("rpc", written)

    def test_cards_kept_apart_are_kept_apart_by_an_override_that_moves_nothing(self):
        written = section(across((SLOW, FAST), (25, 41), (1024, 2048),
                                 pipeline=Pipeline.OFF), named="model-2gpu")

        self.assertEqual(f"{NO_TENSOR}=CUDA1", written["override-tensor"])

    def test_a_slaves_card_is_reached_through_its_worker_and_overrides_nothing(self):
        written = section(across((SLAVE, SLOW, FAST), (24, 15, 27), (2048, 1024, 2048),
                                 pipeline=Pipeline.OFF), named="model-rpc")

        self.assertEqual("worker:50052", written["rpc"])
        self.assertEqual("RPC0,CUDA1,CUDA0", written["device"])
        self.assertNotIn("override-tensor", written)


class TheMicroBatchIsWrittenWhereItIsNotTheSharedOne(unittest.TestCase):
    def test_a_halved_micro_batch_is_written_as_the_size_it_runs_at(self):
        written = section(across((SLOW, FAST), (25, 41), (1024, 2048), halvings=2),
                          named="model-2gpu")

        self.assertEqual("128", written["ubatch-size"])

    def test_the_shared_one_is_not_repeated(self):
        self.assertNotIn("ubatch-size",
                         section(across((SLOW, FAST), (25, 41), (1024, 2048)),
                                 named="model-2gpu"))


class EveryDeviceSaysWhatItWillHold(unittest.TestCase):
    def test_the_requirement_names_every_device_in_order(self):
        text = preset(config(), MACHINE, MEMORY,
                      (placed("model", across((SLAVE, SLOW, FAST), (24, 15, 27),
                                              (2048, 1024, 2048), pipeline=Pipeline.OFF)),))
        said = stated(text)["model-rpc"]

        self.assertEqual(1, len(said))
        self.assertIn(f"{REQUIRED}: {12288 - 2048} MiB on RPC0, {8192 - 1024} MiB on CUDA1, "
                      f"{16303 - 2048} MiB on CUDA0", said[0])


class OneCardOfSeveralIsNamed(unittest.TestCase):
    """A machine with several cards running a profile on one of them has to say which, or
    the router takes the first card it counts."""

    def test_the_section_names_the_card_and_splits_nothing(self):
        written = section(across((FAST,), (66,), (5308,), pipeline=Pipeline.OFF))

        self.assertEqual("CUDA0", written["device"])
        self.assertEqual("none", written["split-mode"])
        for key in ("tensor-split", "rpc", "override-tensor"):
            with self.subTest(key=key):
                self.assertNotIn(key, written)

    def test_the_requirement_names_the_card(self):
        text = preset(config(), MACHINE, MEMORY,
                      (placed("model", across((FAST,), (66,), (5308,),
                                              pipeline=Pipeline.OFF)),))

        self.assertIn(f"{REQUIRED}: {16303 - 5308} MiB on CUDA0", stated(text)["model"][0])

    def test_a_machines_only_card_is_not_named(self):
        written = section(across((FAST,), (66,), (1024,), pipeline=Pipeline.OFF,
                                 among=Among.ONE))

        for key in ("device", "split-mode", "tensor-split"):
            with self.subTest(key=key):
                self.assertNotIn(key, written)


if __name__ == "__main__":
    unittest.main()
