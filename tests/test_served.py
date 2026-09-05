"""Invariants of reading what the router serves out of what it answers.

The command lines here are the ones a running router reports, with the weights moved
somewhere harmless: what is being checked is that the reading survives the real thing,
and a line invented to suit the parser checks nothing.
"""

import unittest

from cm.served import Served, Unusable, display, flag, read, reply_cap
from cm.units import Tokens
from places import somewhere

MODELS = somewhere("models")

# One model as the router reports it, cut to the flags anything here reads.
ARGV = ["llama-server", "--no-webui", "--alias", "qwen3.8-52k-q4-mtp",
        "--ctx-size", "52000", "--cache-ram", "32768",
        "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
        "--flash-attn", "on", "--fit", "off", "--kv-offload",
        "--model", str(MODELS / "unsloth" / "Qwen3.8-27B-GGUF"
                       / "Qwen3.8-27B-UD-IQ4_XS.gguf"),
        "--n-gpu-layers", "99", "--reasoning", "on",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]


def entry(argv=None, **rest) -> dict:
    """One entry of the router's model list, in the shape it arrives in."""
    listed = {"id": "qwen3.8-52k-q4-mtp",
              "object": "model",
              "architecture": {"input_modalities": ["text"],
                               "output_modalities": ["text"]},
              "status": {"args": ARGV if argv is None else argv}}
    listed.update(rest)
    return listed


class AFlagIsReadWithItsValue(unittest.TestCase):
    def test_the_value_follows_the_flag(self):
        self.assertEqual("52000", flag(ARGV, "--ctx-size"))

    def test_a_flag_that_is_not_there(self):
        self.assertEqual("", flag(ARGV, "--n-cpu-moe"))

    def test_a_switch_does_not_swallow_the_next_flag(self):
        """--kv-offload takes no value, and the flag after it is not one."""
        self.assertEqual("", flag(ARGV, "--kv-offload"))

    def test_a_flag_at_the_very_end(self):
        self.assertEqual("", flag(["llama-server", "--ctx-size"], "--ctx-size"))

    def test_the_first_of_two_is_the_answer(self):
        twice = ["--ctx-size", "52000", "--ctx-size", "1"]

        self.assertEqual("52000", flag(twice, "--ctx-size"))


class TheNameSaysWhatTheChoiceIs(unittest.TestCase):
    """A person picks a profile by what it costs them, so the name has to carry it."""

    def test_the_whole_name_of_a_real_profile(self):
        self.assertEqual("Qwen3.8 27B UD-IQ4_XS, q4 cache, MTP (52k)",
                         display(ARGV, Tokens(52000)))

    def test_the_repository_supplies_what_the_file_name_drops(self):
        """A3B is in the directory and not in the file, and it is worth showing."""
        argv = ["--model",
                str(MODELS / "ornith-ai" / "Ornith-1.5-35B-A3B-GGUF"
                    / "Ornith-1.5-35B-Q4_K_M.gguf"),
                "--cache-type-k", "q8_0"]

        self.assertEqual("Ornith 1.5 35B A3B Q4_K_M, q8 cache (131k)",
                         display(argv, Tokens(131072)))

    def test_layers_kept_off_the_card_are_named(self):
        argv = ["40" if one == "99" else one for one in ARGV]

        self.assertIn("part on CPU", display(argv, Tokens(52000)))

    def test_a_model_whole_on_the_card_says_nothing_about_layers(self):
        self.assertNotIn("part on CPU", display(ARGV, Tokens(52000)))

    def test_a_file_with_no_quantisation_in_its_name(self):
        argv = ["--model", str(MODELS / "publisher" / "something" / "model.gguf")]

        self.assertEqual("model (28k)", display(argv, Tokens(28000)))

    def test_a_command_line_naming_no_model(self):
        self.assertEqual("", display(["--ctx-size", "52000"], Tokens(52000)))

    def test_the_window_is_thousands_of_tokens_not_kibibytes(self):
        """28000 has to read back as 28k, not as 27k: windows are chosen in thousands."""
        argv = ["--model", str(MODELS / "publisher" / "repo" / "model.gguf")]

        self.assertIn("(28k)", display(argv, Tokens(28000)))


class OneAnswerMayNotTakeTheWholeWindow(unittest.TestCase):
    def test_a_third_rounded_to_a_power_of_two(self):
        self.assertEqual(Tokens(16384), reply_cap(Tokens(52000)))

    def test_it_never_passes_the_ceiling(self):
        self.assertEqual(Tokens(32768), reply_cap(Tokens(262144)))

    def test_a_short_window_still_reaches_the_floor(self):
        """The extension sizes its own reserve from this, and 8192 is where it starts."""
        self.assertEqual(Tokens(8192), reply_cap(Tokens(28000)))

    def test_a_window_under_the_floor_is_its_own_cap(self):
        """An answer cannot be longer than the window it is written into."""
        self.assertEqual(Tokens(4096), reply_cap(Tokens(4096)))

    def test_a_longer_window_never_allows_a_shorter_answer(self):
        caps = [reply_cap(Tokens(window)) for window in range(1000, 300000, 977)]

        self.assertEqual(caps, sorted(caps))


class AModelIsReadWholeOrNotAtAll(unittest.TestCase):
    def test_everything_comes_from_the_command_line(self):
        self.assertEqual(
            Served(id="qwen3.8-52k-q4-mtp",
                   name="Qwen3.8 27B UD-IQ4_XS, q4 cache, MTP (52k)",
                   window=Tokens(52000), cap=Tokens(16384),
                   modalities=("text",), reasoning=True),
            read(entry()))

    def test_a_model_that_does_not_reason_says_so(self):
        argv = [one if one != "on" else "off" for one in ARGV]

        self.assertFalse(read(entry(argv)).reasoning)

    def test_the_modalities_are_the_ones_reported(self):
        listed = entry()
        listed["architecture"] = {"input_modalities": ["text", "image"]}

        self.assertEqual(("text", "image"), read(listed).modalities)

    def test_a_model_reporting_no_modalities_accepts_text(self):
        listed = entry()
        del listed["architecture"]

        self.assertEqual(("text",), read(listed).modalities)

    def test_a_model_with_no_window_is_left_out_by_name(self):
        argv = [one for one in ARGV if one not in ("--ctx-size", "52000")]

        self.assertEqual(Unusable("qwen3.8-52k-q4-mtp", "reports no window"),
                         read(entry(argv)))

    def test_a_model_reporting_no_command_line(self):
        listed = entry()
        del listed["status"]

        self.assertIsInstance(read(listed), Unusable)

    def test_an_entry_with_no_id_at_all(self):
        listed = entry()
        del listed["id"]

        self.assertEqual(Unusable("", "an entry with no id"), read(listed))


if __name__ == "__main__":
    unittest.main()
