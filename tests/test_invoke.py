"""Invariants of what the estimator is asked.

The estimate has to be about the run the router will actually make. Every flag that
moves the answer is therefore checked to come from the question or from the settings
file, and not from a constant written here.
"""

import unittest

from cm.config import DEFAULT_RUNTIME, Runtime
from cm.invoke import argv, facts_argv
from cm.place import CacheType, ExpertsOnCpu, Question, WholeCard
from cm.units import Layers, Tokens
from places import somewhere

BINARY = somewhere("llama.cpp", "b10448-cuda13.3", "llama-fit-params.exe")
MODEL = somewhere("models", "unsloth", "Qwen3.8.gguf")

WHOLE = Question(Tokens(45000), CacheType.Q8_0, WholeCard())
OFFLOADED = Question(Tokens(131000), CacheType.Q8_0, ExpertsOnCpu(Layers(15)))


def after(flag, line):
    """What was passed with a flag, once."""
    given = [line[i + 1] for i, word in enumerate(line) if word == flag]
    if len(given) != 1:
        raise AssertionError(f"{flag} appears {len(given)} times in {line}")
    return given[0]


class TheQuestionIsWhatIsAsked(unittest.TestCase):
    def test_the_window_is_the_questions_own(self):
        self.assertEqual("45000", after("-c", argv(BINARY, MODEL, WHOLE,
                                                   DEFAULT_RUNTIME)))

    def test_both_halves_of_the_cache_are_the_questions_own(self):
        for cache in CacheType:
            with self.subTest(cache=cache):
                question = Question(Tokens(45000), cache, WholeCard())
                line = argv(BINARY, MODEL, question, DEFAULT_RUNTIME)

                self.assertEqual(cache.value, after("-ctk", line))
                self.assertEqual(cache.value, after("-ctv", line))

    def test_the_file_asked_about_is_the_models(self):
        line = argv(BINARY, MODEL, WHOLE, DEFAULT_RUNTIME)

        self.assertEqual(str(BINARY), line[0])
        self.assertEqual(str(MODEL), after("-m", line))


class ThePlacementIsAskedAboutAsItWillBeRun(unittest.TestCase):
    def test_every_layer_is_on_the_card_either_way(self):
        for question in (WHOLE, OFFLOADED):
            with self.subTest(question=question):
                self.assertEqual("99", after("-ngl",
                                             argv(BINARY, MODEL, question,
                                                  DEFAULT_RUNTIME)))

    def test_experts_are_moved_only_where_the_question_moves_them(self):
        self.assertNotIn("-ncmoe", argv(BINARY, MODEL, WHOLE, DEFAULT_RUNTIME))
        self.assertEqual("15", after("-ncmoe", argv(BINARY, MODEL, OFFLOADED,
                                                    DEFAULT_RUNTIME)))

    def test_the_loader_is_told_not_to_place_anything_itself(self):
        """Left on, it would answer about a placement of its own choosing."""
        line = argv(BINARY, MODEL, WHOLE, DEFAULT_RUNTIME)

        self.assertEqual("off", after("--fit", line))
        self.assertEqual("on", after("--fit-print", line))


class WhatTheRouterWillRunWithIsAskedAbout(unittest.TestCase):
    """Batching and flash attention move the compute buffers by hundreds of megabytes."""

    def test_the_batch_sizes_come_from_the_settings_file(self):
        runtime = Runtime(batch=512, ubatch=128, parallel=4, flash_attn="off")

        line = argv(BINARY, MODEL, WHOLE, runtime)

        self.assertEqual("512", after("-b", line))
        self.assertEqual("128", after("-ub", line))
        self.assertEqual("4", after("-np", line))
        self.assertEqual("off", after("-fa", line))

    def test_a_different_runtime_asks_a_different_question(self):
        loose = Runtime(batch=4096, ubatch=1024, parallel=1, flash_attn="on")

        self.assertNotEqual(argv(BINARY, MODEL, WHOLE, DEFAULT_RUNTIME),
                            argv(BINARY, MODEL, WHOLE, loose))


class TheHeaderIsReadSeparately(unittest.TestCase):
    def test_it_asks_verbosely_because_that_is_where_the_shape_is_printed(self):
        self.assertIn("-v", facts_argv(BINARY, MODEL))

    def test_it_asks_about_the_model_and_places_nothing(self):
        line = facts_argv(BINARY, MODEL)

        self.assertEqual(str(MODEL), after("-m", line))
        self.assertEqual("off", after("--fit", line))


if __name__ == "__main__":
    unittest.main()
