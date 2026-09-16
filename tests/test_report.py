"""Invariants of what calibrate says while it works.

The report is the only place a person sees why a model came out the way it did. Every
number in it is therefore checked to be the one that was decided, and a model that got
nothing is checked to be said out loud rather than left off the list.
"""

import unittest
from pathlib import Path

from cm import place
from cm.config import Model
from cm.machine import Card, SystemMemory
from cm.name import names
from cm.nonempty import NonEmpty
from cm.place import CacheType, ExpertsOnCpu, Settings, WholeCard
from cm.render import Placed
from cm.report import about, closing, opening, system
from cm.units import Layers, Mib, Tokens
from one_card import LAYOUT
from places import somewhere

MODELS = somewhere("models")

CARD = Card("NVIDIA GeForce RTX 5070 Ti", Mib(16303))


def model(key) -> Model:
    return Model(key=key,
                 path=MODELS / f"{key}.gguf",
                 vendor={},
                 allowed=place.EVERYTHING,
                 manual=False)


def settings(ctx=45000, cache=CacheType.Q8_0, head=False,
             placement=WholeCard(), spare=1027) -> Settings:
    return Settings(ctx=Tokens(ctx), cache=cache, head=head,
                    placement=placement, spare=NonEmpty(Mib(spare)),
                    layout=LAYOUT)


def placed(key, chosen, resident=0) -> Placed:
    return Placed(model(key), names(key, chosen), Mib(resident))


class WhatThePlacementsWereComputedAgainstIsSaidFirst(unittest.TestCase):
    def test_the_card_the_room_and_the_reserve_are_all_named(self):
        line = opening(CARD, Mib(15903), Mib(1024))

        self.assertIn(CARD.name, line)
        self.assertIn("16303", line)
        self.assertIn("15903", line)
        self.assertIn("1024", line)

    def test_the_room_and_the_reserve_are_not_read_off_the_card(self):
        """What the driver keeps and what the settings ask for are neither of them."""
        line = opening(CARD, Mib(15903), Mib(2048))

        self.assertIn("15903", line)
        self.assertIn("2048", line)
        self.assertNotIn("11888", line)


class AModelThatGotNothingIsStillReported(unittest.TestCase):
    def test_it_is_named_and_the_reason_is_given(self):
        lines = about(placed("qwen3.8-q4km", ()))

        self.assertEqual("qwen3.8-q4km", lines[0])
        self.assertIn("nothing fits", lines[1])

    def test_it_is_not_silently_left_out(self):
        self.assertTrue(about(placed("qwen3.8-q4km", ())))


class EveryProfileSaysWhatItIs(unittest.TestCase):
    def test_a_line_carries_its_name_window_cache_and_free_memory(self):
        lines = about(placed("qwen3.8", (settings(ctx=45000, spare=1027),)))

        self.assertIn("qwen3.8", lines[1])
        self.assertIn("45000", lines[1])
        self.assertIn("q8_0", lines[1])
        self.assertIn("1027", lines[1])

    def test_offloaded_experts_are_counted_out_loud(self):
        one = settings(ctx=131000, placement=ExpertsOnCpu(Layers(15)))

        self.assertIn("15 layers", about(placed("ornith", (one,)))[1])

    def test_a_model_wholly_on_the_card_says_nothing_about_layers(self):
        self.assertNotIn("layers", about(placed("gemma4", (settings(),)))[1])

    def test_there_is_one_line_per_profile_and_one_for_the_model(self):
        chosen = (settings(ctx=33000, head=True),
                  settings(ctx=45000),
                  settings(ctx=85000, cache=CacheType.Q4_0))

        self.assertEqual(1 + len(chosen), len(about(placed("qwen3.8", chosen))))

    def test_the_shortest_window_is_listed_first(self):
        chosen = (settings(ctx=85000, cache=CacheType.Q4_0),
                  settings(ctx=33000, head=True),
                  settings(ctx=45000))

        lines = about(placed("qwen3.8", chosen))

        self.assertEqual(["33000", "45000", "85000"],
                         [line.split("tokens")[0].split()[-1] for line in lines[1:]])


class WhatWasWrittenIsCountedAtTheEnd(unittest.TestCase):
    def test_the_profiles_and_the_models_served_are_both_counted(self):
        line = closing(Path("llamacpp.models.ini"),
                       (placed("qwen3.8", (settings(ctx=33000, head=True),
                                           settings(ctx=45000))),
                        placed("gemma4", (settings(),)),
                        placed("qwen3.8-q4km", ())))

        self.assertIn("3 profiles", line)
        self.assertIn("2 of 3 models", line)

    def test_the_file_it_names_is_the_one_that_was_written(self):
        path = somewhere("elsewhere", "other.ini")

        self.assertIn(str(path), closing(path, (placed("gemma4", (settings(),)),)))

    def test_a_run_that_placed_nothing_says_so_rather_than_failing(self):
        line = closing(Path("llamacpp.models.ini"), (placed("qwen3.8-q4km", ()),))

        self.assertIn("0 profiles", line)
        self.assertIn("0 of 1 models", line)


class TheMemoryOffTheCardIsSaidOutLoud(unittest.TestCase):
    """Three numbers of one reading: what there is, what the weights keep, what is left."""

    MEMORY = SystemMemory(installed=Mib(65407), resident=Mib(12159),
                          cache=Mib(45056))

    def test_every_figure_is_in_the_line(self):
        line = system(self.MEMORY)

        for figure in ("65407", "12159", "45056"):
            with self.subTest(figure=figure):
                self.assertIn(figure, line)

    def test_a_machine_holding_nothing_off_the_card_still_says_so(self):
        line = system(SystemMemory(installed=Mib(65407), resident=Mib(0),
                                   cache=Mib(57344)))

        self.assertIn("0", line)
        self.assertIn("57344", line)


if __name__ == "__main__":
    unittest.main()
