"""Invariants of reading the settings file.

Every way the file can be wrong ends in one message that says what to fix, and every
message is checked as written: a person reading it is the only diagnosis this program
offers, so an approximate one is no message at all.
"""

import json
import unittest
from pathlib import Path, PureWindowsPath

from cm import config, library, place, workspace
from cm.config import (DEFAULT_CUDA, DEFAULT_KEPT, DEFAULT_RUNTIME, DERIVED, NEUTRAL,
                       Config, ConfigError, Rename, Runtime, Writing, naming, renames,
                       renaming, retuning, unnamed)
from cm.library import Key
from cm.lmstudio import Found, Missing
from cm.machine import Fitted, Fixed, Share
from cm.place import CacheType
from cm.recommended import Recommended, Unknown
from cm.serving import (DEFAULT_HOST, DEFAULT_IDLE, DEFAULT_PORT,
                        DEFAULT_RESIDENT)
from cm.units import Mib, Tokens
from cm.upstream import Cuda
from places import somewhere


def quoted(path: Path) -> str:
    """A path as a settings file holds one.

    In double quotes rather than TOML's literal ones, which cannot hold a quote at all:
    a scratch directory under somebody's name may well have one in it.
    """
    return json.dumps(str(path))


MODELS = somewhere("models")

# Where a model file sits under the library, and what name a file asks the preset be
# written under: both relative, the way both are written in a settings file.
MODEL_FILE = Path("unsloth") / "Qwen3.8.gguf"
PRESET = Path("elsewhere") / "preset.ini"

MODEL_ROOT = f"model_root = {quoted(MODELS)}"
FILE = f"file = {quoted(MODEL_FILE)}"

WHOLE = f"""
{MODEL_ROOT}
preset_path = {quoted(PRESET)}

cuda_version  = '12.4'
keep_releases = 3

listen_host        = '127.0.0.1'
port               = 18099
models_max         = 2
sleep_idle_seconds = 60
log_dir            = 'router-logs'

reserve_mib      = 900
min_ctx_tokens   = 30000
ample_ctx_tokens = 120000

[shared]
flash-attn = 'on'
parallel   = 1
jinja      = true

[models.'qwen3.8']
{FILE}
cache = ['q8_0']
mtp   = false

[models.'qwen3.8'.settings]
temp  = '1.0'
top-k = '20'
"""

BARE = f"""
{MODEL_ROOT}

[models.'qwen3.8']
{FILE}
"""


# The library this machine's LM Studio recorded, as these tests are read: somewhere
# other than MODELS, so a test that takes one for the other fails. Missing by default:
# every file above names a root of its own, so a test that accidentally stopped naming
# one would be refused rather than quietly served from somewhere else.
ELSEWHERE = somewhere("lmstudio", "models")
NO_LIBRARY = Missing(ELSEWHERE)
LIBRARY = Found(ELSEWHERE)


def parse(text, library=NO_LIBRARY) -> Config:
    return config.parse(text, library)


def refused(text, library=NO_LIBRARY) -> str:
    """The one line a settings file was refused with."""
    try:
        parse(text, library)
    except ConfigError as error:
        return str(error)
    raise AssertionError("the file was accepted")


class AFileThatCannotBeActedOnSaysWhy(unittest.TestCase):
    def test_a_models_table_that_is_not_a_table(self):
        self.assertEqual('models must be a table: one [models."name"] entry per model',
                         refused(BARE.split("[models")[0] + "models = 'none'\n"))

    def test_a_hidden_that_is_neither(self):
        self.assertEqual("qwen3.8: hidden must be true or false",
                         refused(BARE + "\nhidden = 'later'\n"))

    def test_a_manual_that_is_neither(self):
        self.assertEqual("qwen3.8: manual must be true or false",
                         refused(BARE + "\nmanual = 'sometimes'\n"))

    def test_a_flag_written_under_the_settings_block_does_nothing_and_says_so(self):
        """TOML gives a bare key to the last header above it, so a flag written under
        [models."x".settings] is a sampler value called hidden that hides nothing."""
        for flag in ("hidden", "manual"):
            with self.subTest(flag=flag):
                text = f"{BARE}\n[models.'qwen3.8'.settings]\n{flag} = true\n"

                self.assertEqual(
                    f"qwen3.8: {flag} is a flag of the entry and not a sampler value, "
                    f'but it is written under [models."qwen3.8".settings], where it '
                    "does nothing. Move it above that line.",
                    refused(text))

    def test_a_model_with_no_file(self):
        self.assertEqual("qwen3.8: file is not set",
                         refused(BARE.replace(FILE, "")))

    def test_a_number_that_is_not_one(self):
        for key in ("reserve_mib", "min_ctx_tokens", "ample_ctx_tokens",
                    "keep_releases", "port", "models_max", "sleep_idle_seconds"):
            with self.subTest(key=key):
                text = BARE.replace("[models", key + " = 'plenty'\n\n[models")

                self.assertEqual(key + " must be a whole number", refused(text))

    def test_a_cuda_version_that_is_not_written_as_one(self):
        """13.3 unquoted is a number, and 13.30 is the same number and another
        directory name. The archives are named with the text, so the text is asked
        for."""
        for written in ("''", "13.3", "'  '", "true"):
            with self.subTest(written=written):
                text = BARE.replace(
                    "[models", f"cuda_version = {written}\n\n[models")

                self.assertEqual(
                    "cuda_version must be written in quotes, and not left empty",
                    refused(text))

    def test_an_empty_list_of_caches(self):
        self.assertEqual("qwen3.8: cache is empty",
                         refused(BARE + "\ncache = []\n"))

    def test_a_cache_no_kernel_exists_for(self):
        self.assertEqual("qwen3.8: unknown cache q3_0",
                         refused(BARE + "\ncache = ['q3_0']\n"))

    def test_a_head_that_is_neither_wanted_nor_not(self):
        self.assertEqual("qwen3.8: mtp must be true or false",
                         refused(BARE + "\nmtp = 'sometimes'\n"))


class HowTheRouterIsRunIsReadTheSameWay(unittest.TestCase):
    """Every part of it has a default, so a settings file that says nothing about the
    router still describes one that can be started."""

    def test_what_the_file_says(self):
        given = parse(WHOLE).serving

        self.assertEqual("127.0.0.1", given.host)
        self.assertEqual(18099, given.port)
        self.assertEqual(2, given.resident)
        self.assertEqual(60, given.idle)

    def test_what_it_does_not_say(self):
        given = parse(BARE).serving

        self.assertEqual(DEFAULT_HOST, given.host)
        self.assertEqual(DEFAULT_PORT, given.port)
        self.assertEqual(DEFAULT_RESIDENT, given.resident)
        self.assertEqual(DEFAULT_IDLE, given.idle)

    def test_the_logs_land_beside_the_settings_file_by_default(self):
        """Relative, like every other place the file names, and already kept out of the
        repository. Not under the releases: a release is replaced, and the log of what
        the router did before that is not part of what was replaced."""
        self.assertEqual(Path("logs"), parse(BARE).serving.logs)

    def test_a_log_directory_the_file_names(self):
        self.assertEqual(Path("router-logs"), parse(WHOLE).serving.logs)

    def test_a_port_nothing_can_listen_on(self):
        for written in ("0", "65536", "-1"):
            with self.subTest(written=written):
                text = BARE.replace("[models", f"port = {written}\n\n[models")

                self.assertEqual("port must be between 1 and 65535", refused(text))

    def test_a_router_that_may_hold_no_model(self):
        self.assertEqual(
            "models_max must be at least 1, or the router may hold no model and "
            "serves nothing",
            refused(BARE.replace("[models", "models_max = 0\n\n[models")))

    def test_an_idle_period_that_runs_backwards(self):
        self.assertEqual(
            "sleep_idle_seconds cannot be negative",
            refused(BARE.replace("[models", "sleep_idle_seconds = -60\n\n[models")))

    def test_an_address_left_empty(self):
        self.assertEqual(
            "listen_host must be written in quotes, and not left empty",
            refused(BARE.replace("[models", "listen_host = ''\n\n[models")))


class WhereTheWeightsAreWhenTheFileDoesNotSay(unittest.TestCase):
    """LM Studio is how these files get onto a machine, and it records where it put them.

    A settings file repeating that path is a file that goes stale when the library
    moves, so leaving the key out is the ordinary case rather than the fallback.
    """

    WITHOUT = BARE.replace(MODEL_ROOT, "")

    def test_the_machine_own_library_is_used(self):
        self.assertEqual(ELSEWHERE, parse(self.WITHOUT, LIBRARY).model_root)

    def test_every_model_path_is_built_from_it(self):
        given = parse(self.WITHOUT, LIBRARY)

        self.assertEqual(ELSEWHERE / MODEL_FILE, given.models[0].path)

    def test_a_named_root_is_not_overruled_by_the_library(self):
        """The file was written by a person; the library was found by a program."""
        self.assertEqual(MODELS, parse(BARE, LIBRARY).model_root)

    def test_no_key_and_no_library_says_both_halves(self):
        self.assertEqual(
            "model_root is not set and there is no LM Studio library at "
            f"{ELSEWHERE}. Set model_root to the directory the weights are under.",
            refused(self.WITHOUT))


class HowMuchMemoryTheCacheMayHaveIsWrittenAsAnyoneWouldWriteIt(unittest.TestCase):
    """32, 32G, 32Gb, 32GiB are one size; 50% is a share; no key at all is neither."""

    def written(self, value):
        text = BARE.replace("[models", f"cache_ram = {value}\n\n[models")
        return parse(text).cache_ram

    def test_a_bare_number_is_gibibytes(self):
        self.assertEqual(Fixed(Mib(32768)), self.written("32"))

    def test_every_spelling_of_the_unit_means_the_same(self):
        for spelling in ("'32'", "'32G'", "'32Gb'", "'32GiB'", "'32 gb'", "'32g'"):
            with self.subTest(written=spelling):
                self.assertEqual(Fixed(Mib(32768)), self.written(spelling))

    def test_a_share_is_kept_as_a_share(self):
        """Of what, this cannot know: the memory is a fact about the machine."""
        self.assertEqual(Share(50), self.written("'50%'"))

    def test_no_key_leaves_it_to_the_machine(self):
        self.assertEqual(Fitted(), parse(BARE).cache_ram)

    def test_a_share_of_more_than_everything(self):
        text = BARE.replace("[models", "cache_ram = '120%'\n\n[models")

        self.assertEqual("cache_ram: a share over 100% is more memory than there is",
                         refused(text))

    def test_something_that_is_not_a_size(self):
        for value in ("'lots'", "'32 kg'", "true", "1.5", "'-4G'", "[]"):
            with self.subTest(written=value):
                text = BARE.replace("[models", f"cache_ram = {value}\n\n[models")

                self.assertEqual("cache_ram: write it as 32, '32G', '32Gb' or '50%'",
                                 refused(text))


class ADerivedKeyIsRefusedRatherThanObeyed(unittest.TestCase):
    """A window quietly overridden is the failure the whole program exists to prevent.

    It would not even look like one: the router starts, serves the model, and holds a
    different window than the card was measured for.
    """

    def test_every_derived_key_is_named_and_refused(self):
        for key in sorted(config.DERIVED):
            with self.subTest(key=key):
                text = BARE + f"\n[models.'qwen3.8'.settings]\n{key} = 'whatever'\n"

                self.assertEqual(f"qwen3.8: {key} is derived; remove it", refused(text))

    def test_the_keys_a_placement_writes_are_all_of_them(self):
        """Nothing calibrate writes into a section may also come from the file."""
        written = {"model", "ctx-size", "fit", "gpu-layers", "cache-type-k",
                   "cache-type-v", "n-cpu-moe", "spec-type", "spec-draft-n-max",
                   "spec-draft-type-k", "spec-draft-type-v",
                   "threads", "threads-batch", "cache-ram", "fit-target",
                   "device", "split-mode", "tensor-split", "ubatch-size", "rpc",
                   "override-tensor"}

        self.assertEqual(written, set(config.DERIVED))


class WhatTheFileSaysIsWhatComesOut(unittest.TestCase):
    def test_every_value_is_carried_over(self):
        given = parse(WHOLE)

        self.assertEqual(MODELS, given.model_root)
        self.assertEqual(Cuda("12.4"), given.cuda)
        self.assertEqual(3, given.keep_releases)
        self.assertEqual(PRESET, given.preset_path)
        self.assertEqual(Mib(900), given.reserve)
        self.assertEqual(Tokens(30000), given.min_ctx)
        self.assertEqual(Tokens(120000), given.ample_ctx)

    def test_a_models_file_is_found_under_the_root(self):
        given = parse(WHOLE)

        self.assertEqual(MODELS / MODEL_FILE, given.models[0].path)

    def test_vendor_settings_pass_through_untouched(self):
        given = parse(WHOLE)

        self.assertEqual({"temp": "1.0", "top-k": "20"}, dict(given.models[0].vendor))

    def test_shared_flags_pass_through_untouched(self):
        given = parse(WHOLE)

        self.assertEqual({"flash-attn": "on", "parallel": 1, "jinja": True},
                         dict(given.shared))

    def test_what_a_person_rules_out_arrives_as_ruled_out(self):
        given = parse(WHOLE)

        self.assertEqual(frozenset({CacheType.Q8_0}), given.models[0].allowed.caches)
        self.assertFalse(given.models[0].allowed.head)

    def test_a_model_set_aside_is_kept_apart_from_the_ones_to_place(self):
        """It is out of the way of everything that places or serves, and still there
        for scan, which is what stops its file being named a second time."""
        given = parse(BARE + "\nhidden = true\n")

        self.assertEqual((), given.models)
        self.assertEqual(("qwen3.8",), tuple(one.key for one in given.withheld))
        self.assertEqual(MODELS / MODEL_FILE, given.withheld[0].path)


class WhatIsNotWrittenDownHasAnAnswerAnyway(unittest.TestCase):
    def test_the_numbers_fall_back_to_the_ones_the_core_carries(self):
        given = parse(BARE)

        self.assertEqual(place.DEFAULT_RESERVE, given.reserve)
        self.assertEqual(place.DEFAULT_MIN_CTX, given.min_ctx)
        self.assertEqual(place.DEFAULT_AMPLE_CTX, given.ample_ctx)

    def test_the_cuda_version_and_how_many_releases_to_keep_have_defaults(self):
        """Neither is a fact about this machine that anything can read off it, and
        both are what the machine this was written on has always used."""
        given = parse(BARE)

        self.assertEqual(DEFAULT_CUDA, given.cuda)
        self.assertEqual(DEFAULT_KEPT, given.keep_releases)

    def test_the_preset_lands_beside_the_settings_by_default(self):
        self.assertEqual(Path("llamacpp.models.ini"), parse(BARE).preset_path)

    def test_a_model_that_rules_out_nothing_may_do_everything(self):
        self.assertEqual(place.EVERYTHING, parse(BARE).models[0].allowed)

    def test_a_file_naming_no_model_names_no_model(self):
        """A machine that has not run scan yet, which is not a machine that is wrong.
        install copies the template and reads the copy in the same run, so refusing
        this would refuse an install that had done nothing."""
        given = parse(BARE.split("[models")[0])

        self.assertEqual((), given.models)
        self.assertEqual((), given.withheld)

    def test_a_model_is_offered_unless_it_is_hidden(self):
        self.assertEqual(("qwen3.8",), tuple(one.key for one in parse(BARE).models))
        self.assertEqual((), parse(BARE).withheld)

    def test_a_file_with_no_shared_block_shares_nothing(self):
        self.assertEqual({}, dict(parse(BARE).shared))


class WhatMovesAnEstimateIsReadOutOfTheSharedBlock(unittest.TestCase):
    """The estimator has to be asked about the run the router will make, not another."""

    def test_the_four_flags_are_taken_as_written(self):
        given = parse(BARE.replace("[models", """[shared]
batch-size  = 512
ubatch-size = 128
parallel    = 4
flash-attn  = 'off'

[models"""))

        self.assertEqual(Runtime(batch=512, ubatch=128, parallel=4, flash_attn="off"),
                         given.runtime)

    def test_a_file_that_names_none_of_them_gets_llama_cpps_own(self):
        self.assertEqual(DEFAULT_RUNTIME, parse(BARE).runtime)

    def test_reading_them_does_not_take_them_away_from_the_router(self):
        given = parse(BARE.replace(
            "[models", "[shared]\nflash-attn = 'off'\n\n[models"))

        self.assertEqual("off", given.shared["flash-attn"])

    def test_a_batch_that_is_not_a_count_is_refused_by_name(self):
        for key in ("batch-size", "ubatch-size", "parallel"):
            for value in ("'many'", "0", "-1", "true", "2.5"):
                with self.subTest(key=key, value=value):
                    text = BARE.replace(
                        "[models",
                        f"[shared]\n{key} = {value}\n\n[models")

                    self.assertEqual(
                        f"shared: {key} must be a whole number above zero",
                        refused(text))


class WhereTheModelsAreCanBeWrittenIntoTheFile(unittest.TestCase):
    """The one answer install asks for on a machine LM Studio is not on.

    It goes into the copy install just made, so what is checked here is that the text
    that comes out is a settings file that parses, and parses to the directory that was
    given -- not that some line was replaced.
    """

    # A directory as awkward as a real one: absolute, with a space and an apostrophe,
    # which any ordinary possessive puts into a folder name. Taken from the system so
    # that the drive and the separators are this system's rather than invented.
    AWKWARD = somewhere("everyone's models", "GGUF")

    def written(self, text, models=AWKWARD) -> Config:
        return parse(config.naming_the_library(text, models), NO_LIBRARY)

    def test_the_template_comes_out_naming_that_directory(self):
        shipped = workspace.template().read_text(encoding="utf-8")

        self.assertEqual(self.AWKWARD, self.written(shipped).model_root)

    def test_a_file_that_names_one_already_is_made_to_name_this_one(self):
        self.assertEqual(self.AWKWARD, self.written(WHOLE).model_root)

    def test_a_file_with_no_such_key_at_all(self):
        """It is prepended, and a top-level key has to come before the first table:
        after one it would be a key of that table and mean something else."""
        self.assertEqual(self.AWKWARD, self.written(BARE.replace(MODEL_ROOT, "")).model_root)

    def test_saying_it_twice_says_it_once(self):
        once = config.naming_the_library(BARE, self.AWKWARD)

        self.assertEqual(once, config.naming_the_library(once, self.AWKWARD))

    def test_nothing_else_in_the_file_is_touched(self):
        given = self.written(WHOLE)

        self.assertEqual(Cuda("12.4"), given.cuda)
        self.assertEqual(18099, given.serving.port)
        self.assertEqual(("qwen3.8",), tuple(one.key for one in given.models))


class AFileTheSettingsDoNotNameIsOneScanHasToWrite(unittest.TestCase):
    """What is in the library and not in the file, matched on the file itself.

    On the file rather than on the key, because the key is the person's to change: an
    entry they renamed still has to count as the entry for its file, or scan writes a
    second one and the model is served twice under two names.
    """

    HERE = PureWindowsPath("unsloth", "Qwen3.8-27B-GGUF", "Qwen3.8-27B-Q4_K_M.gguf")

    def entries(self, text: str = BARE):
        return parse(text).models

    def test_a_file_no_entry_names_is_one_to_write(self):
        self.assertEqual((self.HERE,), unnamed(self.entries(), (self.HERE,), MODELS))

    def test_a_file_an_entry_already_names_is_left_alone(self):
        self.assertEqual((), unnamed(self.entries(), (MODEL_FILE,), MODELS))

    def test_an_entry_that_was_renamed_still_names_its_file(self):
        renamed = BARE.replace("[models.'qwen3.8']", "[models.'whatever-i-call-it']")

        self.assertEqual((), unnamed(self.entries(renamed), (MODEL_FILE,), MODELS))

    def test_an_entry_set_aside_still_names_its_file(self):
        """Which is the whole of why hidden = true exists rather than deleting it."""
        aside = parse(BARE + "\nhidden = true\n")

        self.assertEqual((), unnamed(aside.withheld, (MODEL_FILE,), MODELS))

    def test_the_same_file_written_in_another_case_is_the_same_file(self):
        """A Windows library, where it is one file however it was typed."""
        shouted = PureWindowsPath(str(MODEL_FILE).upper())

        self.assertEqual((), unnamed(self.entries(), (shouted,), MODELS))


QWEN_FILE = PureWindowsPath("unsloth", "Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ4_XS.gguf")

# What a row in the repository says about that model, as scan hands it over. Two keys
# and no more, because a page states what it has an opinion about: what it leaves open
# is what the floor is for.
PUBLISHED = Recommended(settings={"temp": "0.6", "top-k": "20"},
                        source="https://example.invalid/card",
                        pattern="qwen3.8-27b")

# What the entry should then run on.
OVER_THE_FLOOR = {**NEUTRAL, **PUBLISHED.settings}


def unadvised(key: str, place: PureWindowsPath) -> Writing:
    """An entry for a model nothing in the repository covers yet."""
    return Writing(key=Key(key), place=place, advice=Unknown(library.stem(place)))


def advised(key: str, place: PureWindowsPath) -> Writing:
    """An entry for a model the repository has a row for."""
    return Writing(key=Key(key), place=place, advice=PUBLISHED)


class ScanWritesEntriesAndNothingElse(unittest.TestCase):
    QWEN = unadvised("qwen3.8-27b-ud-iq4xs", QWEN_FILE)

    def test_nothing_to_add_leaves_the_text_exactly_as_it_was(self):
        self.assertEqual(BARE, naming(BARE, ()))

    def test_what_was_there_is_still_there_word_for_word(self):
        """Their comments, their order, their own numbers. A writer that reproduced the
        file would be a second parser to keep in step."""
        self.assertTrue(naming(BARE, (self.QWEN,)).startswith(BARE.rstrip("\n")))

    def test_the_entry_reads_back_as_the_model_it_was_written_for(self):
        given = parse(naming(BARE, (self.QWEN,)))
        written = [one for one in given.models if one.key == self.QWEN.key]

        self.assertEqual(1, len(written))
        self.assertEqual(MODELS / self.QWEN.place, written[0].path)

    def test_a_name_carrying_a_version_number_is_one_key_and_not_two_tables(self):
        """qwen3.8 unquoted is a table qwen3 holding a table 8, and the model is lost
        somewhere inside it."""
        given = parse(naming(BARE, (self.QWEN,)))

        self.assertIn(self.QWEN.key, tuple(one.key for one in given.models))

    def test_a_place_holding_an_apostrophe_is_still_written_readably(self):
        """TOML's literal strings, which keep a Windows path as it looks, cannot hold
        an apostrophe at all -- so that one file takes the other kind of string."""
        awkward = unadvised("m-q4km",
                            PureWindowsPath("someone's models", "r", "m-Q4_K_M.gguf"))
        given = parse(naming(BARE, (awkward,)))

        self.assertEqual(MODELS / awkward.place,
                         [one for one in given.models if one.key == "m-q4km"][0].path)

    def test_an_entry_the_repository_covers_starts_on_what_it_publishes(self):
        given = parse(naming(BARE, (advised("qwen3.8-27b-ud-iq4xs", QWEN_FILE),)))
        written = [one for one in given.models if one.key.endswith("iq4xs")][0]

        self.assertEqual(OVER_THE_FLOOR, dict(written.vendor))

    def test_a_page_fills_in_only_what_it_has_an_opinion_about(self):
        """Saying nothing about min-p is not recommending llama.cpp's 0.05. So the row
        wins where it speaks and the floor holds everywhere else, and the row can be
        read against the page it names without the floor being mixed into it."""
        given = parse(naming(BARE, (advised("q", QWEN_FILE),)))
        written = [one for one in given.models if one.key == "q"][0]

        self.assertEqual("0.6", written.vendor["temp"])
        self.assertEqual(NEUTRAL["min-p"], written.vendor["min-p"])
        self.assertEqual(frozenset(NEUTRAL) | frozenset(PUBLISHED.settings),
                         frozenset(written.vendor))

    def test_an_entry_nothing_covers_starts_on_neutral_values(self):
        given = parse(naming(BARE, (self.QWEN,)))
        written = [one for one in given.models if one.key == self.QWEN.key][0]

        self.assertEqual(dict(NEUTRAL), dict(written.vendor))

    def test_where_the_numbers_came_from_is_written_beside_them(self):
        """A block saying what it runs on and not where that was decided is one nobody
        can check."""
        self.assertIn(PUBLISHED.source,
                      naming(BARE, (advised("q", QWEN_FILE),)))

    def test_what_it_writes_is_never_a_key_calibrate_works_out_itself(self):
        """A placement written into an entry is refused by the reader, so a derived key
        among these would be a scan that leaves the file unreadable."""
        self.assertEqual(frozenset(), frozenset(NEUTRAL) & DERIVED)

    def test_every_model_written_at_once_arrives_at_once(self):
        second = unadvised("gemma-4-12b-q4",
                           PureWindowsPath("p", "gemma-4-12B-GGUF",
                                           "gemma-4-12B-Q4_0.gguf"))
        given = parse(naming(BARE, (self.QWEN, second)))

        self.assertEqual({"qwen3.8", self.QWEN.key, second.key},
                         {one.key for one in given.models})


class TheNameForcedOnAnEntryIsTheOneTheLibraryBuilds(unittest.TestCase):
    """--force keys an entry by its file, and the names are built for the whole set.

    An entry whose key is already that name is not a rename. An entry this may not key
    -- marked manual, or holding a file the library cannot name -- keeps its key, and
    that key is spoken for: a name built for another file may not land on it.
    """

    IQ4XS = PureWindowsPath("unsloth", "Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ4_XS.gguf")
    Q4KM = PureWindowsPath("lmstudio-community", "Qwen3.8-27B-GGUF",
                           "Qwen3.8-27B-Q4_K_M.gguf")
    PROJECTOR = PureWindowsPath("unsloth", "Qwen3.8-27B-GGUF", "mmproj-Qwen3.8-27B.gguf")

    def entry(self, key, place, *lines) -> str:
        return (f"[models.{key!r}]\nfile = {str(place)!r}\n"
                + "".join(f"{line}\n" for line in lines))

    def asked(self, *entries) -> tuple:
        given = parse(MODEL_ROOT + "\n\n" + "\n".join(entries))
        return renames(given.models + given.withheld, MODELS)

    def test_an_entry_keyed_by_hand_takes_the_built_name(self):
        self.assertEqual((Rename(was=Key("short"), now=Key("qwen3.8-27b-ud-iq4xs")),),
                         self.asked(self.entry("short", self.IQ4XS)))

    def test_an_entry_already_keyed_that_way_is_not_a_rename(self):
        self.assertEqual((), self.asked(self.entry("qwen3.8-27b-ud-iq4xs", self.IQ4XS)))

    def test_an_entry_marked_manual_keeps_its_key(self):
        self.assertEqual((), self.asked(self.entry("short", self.IQ4XS,
                                                   "manual = true")))

    def test_a_manual_entry_holding_a_built_name_is_stepped_around(self):
        """Its key cannot move, so the name built for the other file goes further out
        rather than onto it."""
        asked = self.asked(self.entry("qwen3.8-27b-ud-iq4xs", self.Q4KM, "manual = true"),
                           self.entry("short", self.IQ4XS))

        self.assertEqual((Rename(was=Key("short"),
                                 now=Key("unsloth-qwen3.8-27b-ud-iq4xs")),), asked)

    def test_an_entry_holding_a_file_that_is_not_a_model_is_stepped_around(self):
        """A projector sits beside a model rather than being one, so the library names
        it nothing -- and the name built for the model may not land on its key."""
        asked = self.asked(self.entry("qwen3.8-27b-ud-iq4xs", self.PROJECTOR),
                           self.entry("short", self.IQ4XS))

        self.assertEqual((Rename(was=Key("short"),
                                 now=Key("unsloth-qwen3.8-27b-ud-iq4xs")),), asked)

    def test_an_entry_whose_file_is_outside_the_library_keeps_its_key(self):
        outside = somewhere("elsewhere", "Qwen3.8-27B-UD-IQ4_XS.gguf")

        self.assertEqual((), self.asked(self.entry("short", outside)))


class ScanRenamesAnEntryByItsHeadersAndNothingElse(unittest.TestCase):
    """--force keys an entry the way the library names its file today.

    A rename says what a model is called here and nothing else: the file the entry
    names, the values under it and whatever a person wrote around it all stay where they
    were. An entry this cannot rewrite whole keeps the key it has and is named -- a key
    rewritten in one header and left in another declares two models where there was one,
    and TOML reads such a file as nothing at all.
    """

    RENAME = Rename(was=Key("qwen3.8"), now=Key("qwen3.8-27b-ud-iq4xs"))

    def written(self, text: str) -> str:
        return renaming(text, (self.RENAME,)).text

    def left(self, text: str) -> tuple:
        return renaming(text, (self.RENAME,)).untouched

    def test_nothing_to_rename_leaves_the_text_exactly_as_it_was(self):
        self.assertEqual(BARE, renaming(BARE, ()).text)

    def test_the_entry_answers_to_the_name_the_library_builds(self):
        given = parse(self.written(BARE))

        self.assertEqual([self.RENAME.now], [one.key for one in given.models])

    def test_the_file_it_names_is_the_file_it_named(self):
        given = parse(self.written(BARE))

        self.assertEqual(MODELS / MODEL_FILE, given.models[0].path)

    def test_its_settings_block_is_carried_over_with_it(self):
        given = parse(self.written(BARE + "[models.'qwen3.8'.settings]\ntemp = '0.6'\n"))

        self.assertEqual("0.6", given.models[0].vendor["temp"])

    def test_what_a_person_wrote_around_it_survives(self):
        note = "# fetched by hand, do not throw away\n"
        written = self.written(BARE + note)

        self.assertIn(note, written)
        self.assertIn("qwen3.8-27b-ud-iq4xs", written)

    def test_an_entry_that_is_not_there_is_left_alone_and_named(self):
        gone = Rename(was=Key("nowhere"), now=Key("somewhere"))
        left = renaming(BARE, (gone,))

        self.assertEqual(BARE, left.text)
        self.assertEqual((gone.was,), tuple(one.key for one in left.untouched))

    def test_an_entry_that_opens_its_settings_twice_is_left_alone(self):
        twice = (BARE + "[models.'qwen3.8'.settings]\ntemp = '0.6'\n"
                 + "[models.'qwen3.8'.settings]\ntop-k = '20'\n")

        self.assertEqual(twice, self.written(twice))
        self.assertEqual((self.RENAME.was,), tuple(one.key for one in self.left(twice)))

    def test_an_entry_carrying_a_table_of_its_own_is_left_alone(self):
        """One this does not know to rewrite. Renaming the two headers it does know
        would leave that table under the old key, which is a second model."""
        besides = BARE + "[models.'qwen3.8'.notes]\nwhy = 'the fast one'\n"

        self.assertEqual(besides, self.written(besides))
        self.assertEqual((self.RENAME.was,), tuple(one.key for one in self.left(besides)))

    def test_a_name_another_entry_holds_is_taken_once_that_entry_moves(self):
        """The file keyed 'a' is named 'a-q4km', and the name 'a' belongs to the other
        file. Writing that one first would declare 'a' twice, so it waits a round."""
        two = (MODEL_ROOT + "\n\n[models.'a']\nfile = 'p\\r\\A-Q4_K_M.gguf'\n"
               + "\n[models.'b']\nfile = 'p\\r\\B-Q4_K_M.gguf'\n")
        done = renaming(two, (Rename(was=Key("b"), now=Key("a")),
                              Rename(was=Key("a"), now=Key("a-q4km"))))

        self.assertEqual((), done.untouched)
        self.assertEqual({"a-q4km", "a"}, {one.key for one in parse(done.text).models})

    def test_two_entries_holding_each_other_s_names_are_both_left_alone(self):
        """A ring nothing can be written out of: writing either one first would declare
        a key twice, so neither is written and both are named."""
        two = (MODEL_ROOT + "\n\n[models.'a']\nfile = 'p\\r\\A-Q4_K_M.gguf'\n"
               + "\n[models.'b']\nfile = 'p\\r\\B-Q4_K_M.gguf'\n")
        done = renaming(two, (Rename(was=Key("a"), now=Key("b")),
                              Rename(was=Key("b"), now=Key("a"))))

        self.assertEqual(two, done.text)
        self.assertEqual({"a", "b"}, {one.key for one in done.untouched})

    def test_another_entry_is_not_touched(self):
        two = BARE + "[models.'gemma4-12b']\n" + FILE + "\n"

        self.assertIn("[models.'gemma4-12b']", self.written(two))
        self.assertEqual({"qwen3.8-27b-ud-iq4xs", "gemma4-12b"},
                         {one.key for one in parse(self.written(two)).models})


class ScanBringsAnEntryUpToDateAndTouchesNothingElse(unittest.TestCase):
    """The settings block belongs to scan and the rest of the file belongs to the person.

    Which is the whole bargain: the numbers follow this repository so a correction is
    made once, and everything written around them -- notes, order, the other keys --
    survives being brought up to date.
    """

    def written(self, text: str) -> str:
        return retuning(text, (advised("qwen3.8", MODEL_FILE),)).text

    def left(self, text: str) -> tuple:
        return retuning(text, (advised("qwen3.8", MODEL_FILE),)).untouched

    def test_nothing_to_retune_leaves_the_text_exactly_as_it_was(self):
        self.assertEqual(BARE, retuning(BARE, ()).text)

    def test_an_entry_with_no_block_of_its_own_is_given_one(self):
        given = parse(self.written(BARE))

        self.assertEqual(OVER_THE_FLOOR, dict(given.models[0].vendor))

    def test_an_entry_that_has_one_is_brought_up_to_date(self):
        stale = f"{BARE}\n[models.'qwen3.8'.settings]\ntemp = '9.9'\nmin-p = '0.4'\n"
        given = parse(self.written(stale))

        self.assertEqual(OVER_THE_FLOOR, dict(given.models[0].vendor))

    def test_doing_it_twice_changes_nothing_the_second_time(self):
        once = self.written(BARE)

        self.assertEqual(once, self.written(once))

    def test_a_note_written_above_the_block_is_the_person_s_and_survives(self):
        theirs = f"{BARE}\n# why this one is odd\n[models.'qwen3.8'.settings]\nt = '1'\n"

        self.assertIn("# why this one is odd", self.written(theirs))

    def test_a_note_introducing_the_next_entry_is_not_swallowed(self):
        """It sits after the last value of one block and before the next header, which
        is exactly the span a careless rewrite takes with it."""
        two = (f"{BARE}\n[models.'qwen3.8'.settings]\ntemp = '9.9'\n\n"
               "# what the next one is for\n[models.'other']\n"
               f"{FILE}\n")
        given = self.written(two)

        self.assertIn("# what the next one is for", given)
        self.assertEqual({"qwen3.8", "other"}, {one.key for one in parse(given).models})

    def test_the_other_keys_of_the_entry_are_left_where_they_are(self):
        flagged = BARE + "\nhidden = true\ncache = ['q8_0']\n"
        given = parse(self.written(flagged))

        self.assertEqual((), given.models)
        self.assertEqual(frozenset({CacheType.Q8_0}), given.withheld[0].allowed.caches)

    def test_a_key_quoted_the_other_way_is_the_same_entry(self):
        """A file may write a key in either kind of quotes, and both name one model."""
        doubled = BARE.replace("[models.'qwen3.8']", '[models."qwen3.8"]')
        given = parse(self.written(doubled))

        self.assertEqual(OVER_THE_FLOOR, dict(given.models[0].vendor))

    def test_an_entry_the_file_does_not_name_leaves_the_file_alone(self):
        """Half-editing somebody's file is worse than not editing it."""
        given = retuning(BARE, (advised("absent", MODEL_FILE),))

        self.assertEqual(BARE, given.text)
        self.assertEqual(("absent",), tuple(one.key for one in given.untouched))


class AnEntryWrittenAWayThisCannotRewriteIsLeftAloneAndSaidSo(unittest.TestCase):
    """A settings block is brought up to date by replacing the lines under its header.
    An entry that writes its settings some other way keeps every one of them, so
    writing a block for it as well would declare the same table twice -- and TOML reads
    a file that does that as no file at all.

    So the whole of somebody's settings would stop being readable, on the run that was
    meant to bring one entry up to date. Every way of writing them that this cannot
    replace is therefore left exactly as it is, and named in what the run reports.
    """

    def retuned(self, text: str):
        return retuning(text, (advised("qwen3.8", MODEL_FILE),))

    def assertLeftAlone(self, text: str) -> None:
        given = self.retuned(text)

        self.assertEqual(text, given.text)
        self.assertEqual(("qwen3.8",), tuple(one.key for one in given.untouched))
        self.assertIsInstance(parse(given.text), Config)

    def test_a_header_with_a_comment_after_it_is_still_that_entry_s_block(self):
        """The pattern wants the line to end at the bracket, so this is a block it does
        not see -- and the entry would be handed a second one."""
        self.assertLeftAlone(f"{BARE}\n[models.'qwen3.8'.settings]  # mine\n"
                             "temp = '9.9'\n")

    def test_settings_written_as_an_inline_table(self):
        self.assertLeftAlone(BARE.replace(FILE, f"{FILE}\nsettings = {{ temp = '9.9' }}"))

    def test_settings_written_as_a_dotted_key(self):
        self.assertLeftAlone(BARE.replace(FILE, f"{FILE}\nsettings.temp = '9.9'"))

    def test_a_block_written_after_some_other_table(self):
        """Valid TOML, and out of the run of lines this replaces: the entry ends at the
        next table that is not its own."""
        self.assertLeftAlone(f"{BARE}\n[shared]\njinja = true\n\n"
                             "[models.'qwen3.8'.settings]\ntemp = '9.9'\n")

    def test_what_it_says_is_what_a_person_can_act_on(self):
        given = self.retuned(BARE.replace(FILE, f"{FILE}\nsettings.temp = '9.9'"))

        self.assertIn("settings", given.untouched[0].why)

    def test_the_entry_beside_it_is_still_brought_up_to_date(self):
        """One entry this cannot rewrite is not a run that stops: the others are what
        the person asked for and they are written."""
        two = (f"{BARE}\nsettings.temp = '9.9'\n\n[models.'other']\n{FILE}\n")
        given = retuning(two, (advised("qwen3.8", MODEL_FILE),
                               advised("other", MODEL_FILE)))

        self.assertEqual(("qwen3.8",), tuple(one.key for one in given.untouched))
        self.assertEqual(OVER_THE_FLOOR,
                         dict(parse(given.text).models[1].vendor))


class TheTemplateInThisRepositoryIsWhatInstallCopies(unittest.TestCase):
    """It is what a machine with no settings file is given. Not a check of any number
    in it.

    install copies it and reads the copy in the same run, so a template that has to be
    edited before it parses is an install that fails on a machine that has done nothing
    wrong. That is what this holds down.
    """

    def shipped(self) -> str:
        return workspace.template().read_text(encoding="utf-8")

    def test_it_parses_as_it_ships(self):
        given = parse(self.shipped(), LIBRARY)

        self.assertIsInstance(given, Config)

    def test_it_names_no_model(self):
        """scan writes the entries, off the library of the machine it lands on. A
        template carrying one would name a file nobody has, and be a model that has to
        be deleted before the first calibrate says anything true."""
        given = parse(self.shipped(), LIBRARY)

        self.assertEqual((), given.models)
        self.assertEqual((), given.withheld)

    def test_it_says_nothing_about_where_llama_cpp_is(self):
        """The releases are in .llamacpp beside it, and a key naming a directory would
        be read by nothing and believed by whoever wrote it."""
        self.assertNotIn("llamacpp_root", self.shipped())

    def test_it_leaves_the_weights_to_the_machine(self):
        """model_root still commented out: the library LM Studio recorded is where they are."""
        self.assertEqual(ELSEWHERE, parse(self.shipped(), LIBRARY).model_root)


if __name__ == "__main__":
    unittest.main()
