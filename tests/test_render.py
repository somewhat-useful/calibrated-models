"""Invariants of the preset file as written.

The strongest of these is that the text reads back: it is parsed again with a plain ini
reader and compared against what went in. A preset the router cannot read is not a
preset, and nothing else here would notice.
"""

import configparser
import unittest
from pathlib import Path

from cm import place, render
from cm.config import (DEFAULT_CUDA, DEFAULT_KEPT, DEFAULT_RUNTIME, Config,
                       ConfigError, Model)
from cm.machine import Card, Core, Fitted, Machine, system_memory, threads
from cm.name import names
from cm.nonempty import NonEmpty
from cm.place import CacheType, ExpertsOnCpu, Settings, WholeCard
from cm.render import REQUIRED, Placed, preset
from cm.serving import (DEFAULT_HOST, DEFAULT_IDLE, DEFAULT_PORT,
                        DEFAULT_RESIDENT, Serving)
from cm.units import Layers, Mib, Tokens
from one_card import LAYOUT, installed, shown
from places import somewhere

MODELS = somewhere("models")
LLAMACPP = somewhere("llama.cpp")

MACHINE = Machine(cards=installed(Card("NVIDIA GeForce RTX 5070 Ti", Mib(16303))),
                  ram=Mib(65407),
                  cores=tuple([Core(1, 2)] * 8 + [Core(0, 1)] * 8))


# What the machine settled on for the prompt cache. Worked out here the way calibrate
# works it out, so a change to that rule shows up in the file these tests read.
MEMORY = system_memory(Fitted(), MACHINE, Mib(12159))

# The router as the settings file leaves it: nothing in the preset follows from it.
SERVING = Serving(host=DEFAULT_HOST, port=DEFAULT_PORT, resident=DEFAULT_RESIDENT,
                  idle=DEFAULT_IDLE, logs=LLAMACPP / "logs")


def config(shared=None, reserve=Mib(1024)) -> Config:
    return Config(model_root=MODELS,
                  cuda=DEFAULT_CUDA,
                  keep_releases=DEFAULT_KEPT,
                  preset_path=Path("llamacpp.models.ini"),
                  serving=SERVING,
                  reserve=reserve,
                  min_ctx=Tokens(25000),
                  ample_ctx=Tokens(100000),
                  cache_ram=Fitted(),
                  runtime=DEFAULT_RUNTIME,
                  shared=dict(shared or {}),
                  models=(),
                  withheld=())


def model(key, vendor=None) -> Model:
    return Model(key=key,
                 path=MODELS / f"{key}.gguf",
                 vendor=dict(vendor or {}),
                 allowed=place.EVERYTHING,
                 manual=False)


def settings(ctx, cache=CacheType.Q8_0, head=False, placement=None,
             spare=1024) -> Settings:
    return Settings(ctx=Tokens(ctx),
                    cache=cache,
                    head=head,
                    placement=placement or WholeCard(),
                    spare=NonEmpty(Mib(spare)),
                    layout=LAYOUT)


def placed(key, *chosen, vendor=None, resident=0) -> Placed:
    return Placed(model(key, vendor), names(key, chosen, shown(chosen)), Mib(resident))


DENSE = placed("qwen3.8",
               settings(33000, CacheType.Q8_0, head=True),
               settings(45000, CacheType.Q8_0),
               settings(62000, CacheType.Q4_0, head=True),
               settings(85000, CacheType.Q4_0))

MIXTURE = placed("ornith-1.5-35b",
                 settings(131000, placement=ExpertsOnCpu(Layers(15))))


def read(text):
    """The preset as a router reads it: what stands before the sections, and them."""
    preamble, _, rest = text.partition("\n\n[")
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string("[" + rest)
    return preamble, parser


def stated(text):
    """What each section says it will hold, by section name. Comments and all."""
    found = {}
    name = "before any section"
    for line in text.splitlines():
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1]
            found[name] = []
        elif line.startswith(f"; {REQUIRED}"):
            found[name].append(line)

    return found


class TheFileHasTheShapeARouterExpects(unittest.TestCase):
    def test_it_reads_back_as_an_ini(self):
        _, parsed = read(preset(config(), MACHINE, MEMORY, (DENSE, MIXTURE)))

        self.assertIn("*", parsed)
        self.assertIn("ornith-1.5-35b", parsed)
        self.assertEqual("131000", parsed["ornith-1.5-35b"]["ctx-size"])

    def test_the_version_stands_before_the_first_section(self):
        preamble, _ = read(preset(config(), MACHINE, MEMORY, (DENSE,)))

        self.assertTrue(preamble.endswith("version = 1"))

    def test_what_stands_before_the_version_is_all_comment(self):
        preamble, _ = read(preset(config(), MACHINE, MEMORY, (DENSE,)))

        for line in preamble.splitlines()[:-1]:
            self.assertTrue(line == "" or line.startswith(";"), line)

    def test_the_shared_section_comes_first_and_the_rest_are_sorted(self):
        _, parsed = read(preset(config(), MACHINE, MEMORY, (MIXTURE, DENSE)))

        first, *rest = parsed.sections()
        self.assertEqual("*", first)
        self.assertEqual(sorted(rest), rest)

    def test_model_is_the_first_key_of_every_section_that_serves_one(self):
        text = preset(config(), MACHINE, MEMORY, (DENSE, MIXTURE))

        for block in text.split("\n\n")[3:]:
            name, *body = block.splitlines()
            keys = [line for line in body if not line.startswith(";")]
            with self.subTest(section=name):
                self.assertTrue(keys[0].startswith("model = "), keys[0])

    def test_every_other_key_is_in_alphabetical_order(self):
        text = preset(config(), MACHINE, MEMORY, (DENSE, MIXTURE))

        for block in text.split("\n\n")[2:]:
            name, *body = block.splitlines()
            keys = [line.split(" = ")[0]
                    for line in body if not line.startswith(";")]
            keys = [key for key in keys if key != "model"]
            with self.subTest(section=name):
                self.assertEqual(sorted(keys), keys)


class EveryProfileBecomesOneSection(unittest.TestCase):
    def test_each_profile_appears_once_under_its_own_name(self):
        _, parsed = read(preset(config(), MACHINE, MEMORY, (DENSE, MIXTURE)))

        self.assertEqual(["*", "ornith-1.5-35b", "qwen3.8-33k",
                          "qwen3.8-45k-nomtp", "qwen3.8-62k-q4",
                          "qwen3.8-85k-q4-nomtp"],
                         parsed.sections())

    def test_a_model_that_did_not_fit_contributes_nothing(self):
        _, parsed = read(preset(config(), MACHINE, MEMORY,
                                (DENSE, Placed(model("qwen3.8-q4km"), (), Mib(0)))))

        self.assertNotIn("qwen3.8-q4km", parsed.sections())

    def test_two_models_claiming_one_name_are_refused(self):
        """A key may be anything, so a key may be another model's profile name."""
        clash = placed("qwen3.8-45k-nomtp", settings(45000))

        with self.assertRaises(ConfigError) as refusal:
            preset(config(), MACHINE, MEMORY, (DENSE, clash))

        self.assertEqual("duplicate section: qwen3.8-45k-nomtp", str(refusal.exception))


class ThePlacementIsWrittenOutInFull(unittest.TestCase):
    """A section says how it runs. Editing the shared block must not move a window."""

    def test_the_window_and_the_cache_are_the_profiles_own(self):
        _, parsed = read(preset(config(), MACHINE, MEMORY, (DENSE,)))

        self.assertEqual("85000", parsed["qwen3.8-85k-q4-nomtp"]["ctx-size"])
        self.assertEqual("q4_0", parsed["qwen3.8-85k-q4-nomtp"]["cache-type-k"])
        self.assertEqual("q4_0", parsed["qwen3.8-85k-q4-nomtp"]["cache-type-v"])
        self.assertEqual("q8_0", parsed["qwen3.8-45k-nomtp"]["cache-type-k"])

    def test_every_section_pins_what_the_loader_would_otherwise_decide(self):
        _, parsed = read(preset(config(), MACHINE, MEMORY, (DENSE, MIXTURE)))

        for name in parsed.sections()[1:]:
            with self.subTest(section=name):
                self.assertEqual("off", parsed[name]["fit"])
                self.assertEqual("99", parsed[name]["gpu-layers"])
                self.assertIn("ctx-size", parsed[name])
                self.assertIn("cache-type-k", parsed[name])
                self.assertIn("cache-type-v", parsed[name])

    def test_experts_are_named_where_they_sit_on_the_cpu_and_nowhere_else(self):
        _, parsed = read(preset(config(), MACHINE, MEMORY, (DENSE, MIXTURE)))

        self.assertEqual("15", parsed["ornith-1.5-35b"]["n-cpu-moe"])
        for name in ("qwen3.8-33k", "qwen3.8-45k-nomtp", "qwen3.8-85k-q4-nomtp"):
            with self.subTest(section=name):
                self.assertNotIn("n-cpu-moe", parsed[name])

    def test_the_draft_head_is_configured_where_it_runs_and_nowhere_else(self):
        _, parsed = read(preset(config(), MACHINE, MEMORY, (DENSE, MIXTURE)))

        drafting = parsed["qwen3.8-33k"]
        self.assertEqual("draft-mtp", drafting["spec-type"])
        self.assertEqual(str(render.DRAFT_LOOKAHEAD), drafting["spec-draft-n-max"])

        for name in ("qwen3.8-45k-nomtp", "ornith-1.5-35b"):
            with self.subTest(section=name):
                self.assertNotIn("spec-type", parsed[name])
                self.assertNotIn("spec-draft-n-max", parsed[name])

    def test_the_head_holds_its_history_at_the_profiles_own_precision(self):
        """One cache precision per profile, and it is the one the placement paid for.

        The head is one more layer of the same conversation. Charging its history at one
        precision and allocating it at another is an error whichever way round it is.
        """
        _, parsed = read(preset(config(), MACHINE, MEMORY, (DENSE,)))

        for name in ("qwen3.8-33k", "qwen3.8-62k-q4"):
            with self.subTest(section=name):
                cache = parsed[name]["cache-type-k"]

                self.assertEqual(cache, parsed[name]["spec-draft-type-k"])
                self.assertEqual(cache, parsed[name]["spec-draft-type-v"])

    def test_the_two_mtp_profiles_do_not_hold_it_at_the_same_precision(self):
        """Without this the test above passes on a constant that happens to match one."""
        _, parsed = read(preset(config(), MACHINE, MEMORY, (DENSE,)))

        self.assertNotEqual(parsed["qwen3.8-33k"]["spec-draft-type-k"],
                            parsed["qwen3.8-62k-q4"]["spec-draft-type-k"])

    def test_the_file_a_section_serves_is_the_one_the_settings_named(self):
        _, parsed = read(preset(config(), MACHINE, MEMORY, (DENSE,)))

        self.assertEqual(str(MODELS / "qwen3.8.gguf"),
                         parsed["qwen3.8-45k-nomtp"]["model"])


class TheMachineAndThePersonBothSpeak(unittest.TestCase):
    def test_the_machine_numbers_are_written_where_the_file_is_silent(self):
        _, parsed = read(preset(config(), MACHINE, MEMORY, (DENSE,)))

        self.assertEqual(str(MEMORY.cache), parsed["*"]["cache-ram"])
        self.assertEqual(str(threads(MACHINE.cores)), parsed["*"]["threads"])
        self.assertEqual(str(threads(MACHINE.cores)), parsed["*"]["threads-batch"])

    def test_a_hand_written_key_wins_over_the_reading(self):
        given = config(shared={"threads": 6, "cache-ram": 8192})

        _, parsed = read(preset(given, MACHINE, MEMORY, (DENSE,)))

        self.assertEqual("6", parsed["*"]["threads"])
        self.assertEqual("8192", parsed["*"]["cache-ram"])

    def test_the_reserve_becomes_what_an_unprofiled_model_places_against(self):
        _, parsed = read(preset(config(reserve=Mib(900)), MACHINE, MEMORY, (DENSE,)))

        self.assertEqual("900", parsed["*"]["fit-target"])

    def test_shared_flags_reach_the_shared_section_verbatim(self):
        given = config(shared={"flash-attn": "on", "jinja": True, "parallel": 1})

        _, parsed = read(preset(given, MACHINE, MEMORY, (DENSE,)))

        self.assertEqual("on", parsed["*"]["flash-attn"])
        self.assertEqual("true", parsed["*"]["jinja"])
        self.assertEqual("1", parsed["*"]["parallel"])

    def test_vendor_settings_reach_every_profile_of_their_model(self):
        drafts = placed("qwen3.8",
                        settings(33000, head=True),
                        settings(45000),
                        vendor={"temp": "1.0", "top-k": "20"})

        _, parsed = read(preset(config(), MACHINE, MEMORY, (drafts,)))

        for name in ("qwen3.8-33k", "qwen3.8-45k-nomtp"):
            with self.subTest(section=name):
                self.assertEqual("1.0", parsed[name]["temp"])
                self.assertEqual("20", parsed[name]["top-k"])



class EachModelIsLeftTheCacheItsOwnWeightsDoNotTake(unittest.TestCase):
    """The router holds one model at a time, so the room for cached prompts is what
    that model leaves -- not what the heaviest model in the file leaves."""

    HEAVY = placed("ornith-1.5-35b",
                   settings(131000, placement=ExpertsOnCpu(Layers(15))),
                   resident=45440)

    def test_a_section_is_given_what_is_left_after_its_own_model(self):
        _, parsed = read(preset(config(), MACHINE, MEMORY, (DENSE, self.HEAVY)))

        expected = system_memory(Fitted(), MACHINE, Mib(45440)).cache

        self.assertEqual(str(expected), parsed["ornith-1.5-35b"]["cache-ram"])

    def test_a_model_holding_nothing_keeps_the_whole_cache(self):
        _, parsed = read(preset(config(), MACHINE, MEMORY, (DENSE, self.HEAVY)))

        whole = system_memory(Fitted(), MACHINE, Mib(0)).cache

        self.assertEqual(str(whole), parsed["qwen3.8-45k-nomtp"]["cache-ram"])
        self.assertGreater(int(parsed["qwen3.8-45k-nomtp"]["cache-ram"]),
                           int(parsed["ornith-1.5-35b"]["cache-ram"]))

    def test_every_profile_of_one_model_is_given_the_same_room(self):
        _, parsed = read(preset(config(), MACHINE, MEMORY, (DENSE,)))

        given = {parsed[name]["cache-ram"] for name in parsed.sections()
                 if name != "*"}

        self.assertEqual(1, len(given))

    def test_a_size_written_into_the_shared_block_stands_for_every_model(self):
        """It is a judgement about the machine, and a section undoing it would leave
        the person's own number showing in [*] and overruled everywhere it matters."""
        given = config(shared={"cache-ram": 8192})

        _, parsed = read(preset(given, MACHINE, MEMORY, (DENSE, self.HEAVY)))

        self.assertEqual("8192", parsed["*"]["cache-ram"])
        for name in parsed.sections():
            if name == "*":
                continue
            with self.subTest(section=name):
                self.assertNotIn("cache-ram", parsed[name])


class EverySectionSaysWhatItWillHold(unittest.TestCase):
    """The figure `vram` reads back. It is in the file for a person and for that tool,
    and the router is never handed it: a key llama.cpp does not know is a key it could
    one day refuse, and refusing would take the router down over a comment."""

    def test_every_profile_states_it_once_and_the_shared_block_never_does(self):
        said = stated(preset(config(), MACHINE, MEMORY, (DENSE, MIXTURE)))

        self.assertEqual([], said["*"])
        for name in [one for one in said if one != "*"]:
            with self.subTest(section=name):
                self.assertEqual(1, len(said[name]))

    def test_the_figure_is_the_card_less_what_the_placement_leaves_free(self):
        one = placed("gemma4-12b", settings(262000, spare=6018))

        said = stated(preset(config(), MACHINE, MEMORY, (one,)))

        self.assertIn(f"{16303 - 6018} MiB", said["gemma4-12b"][0])

    def test_two_profiles_leaving_different_room_state_different_figures(self):
        """Without this the test above passes on a constant that happens to match."""
        said = stated(preset(config(), MACHINE, MEMORY,
                             (placed("close", settings(45000, spare=1024)),
                              placed("roomy", settings(45000, spare=6018)))))

        self.assertNotEqual(said["close"][0], said["roomy"][0])

    def test_an_ini_reader_is_never_handed_it(self):
        _, parsed = read(preset(config(), MACHINE, MEMORY, (DENSE, MIXTURE)))

        for section in parsed.sections():
            for key, value in parsed[section].items():
                with self.subTest(section=section, key=key):
                    self.assertNotIn(REQUIRED, (key + value).upper())
if __name__ == "__main__":
    unittest.main()
