"""Invariants of rewriting pi's configuration.

The files belong to another program. Two things follow, and both are checked here:
nothing outside this router's provider may change, and a number a person set for their
own reasons stands until it stops working.
"""

import copy
import unittest

from cm.agent import UnknownShape, compacted, empty, pointed
from cm.policy import Bound, Room
from cm.served import Router, Served
from cm.units import Tokens

HERE = Router("gpu-box", 18081)

# What the policy leaves pi on this router: its shortest profile, 28k with an 8192 reply
# cap, is governed at a reserve of 8400 and a kept tail of 6860.
ROOM = Room(reserve=Bound(most=Tokens(8399), take=Tokens(8000)),
            keep=Bound(most=Tokens(6860), take=Tokens(4000)))


def served(id, cap=16384, window=52000) -> Served:
    return Served(id=id, name=f"{id} for people", window=Tokens(window),
                  cap=Tokens(cap), modalities=("text",), reasoning=True)


MODELS = (served("qwen3.8-52k-q4-mtp"), served("gemma4-12b", cap=32768, window=262144))

# Another provider, of the kind this program knows nothing about. It has to come back
# exactly as it went in.
STRANGER = {"api": "anthropic", "apiKey": "somebody-elses-key",
            "baseUrl": "https://api.example.invalid/v1",
            "models": [{"id": "something", "contextWindow": 200000}],
            "aSettingThisHasNeverHeardOf": True}


def config(providers=None) -> dict:
    """pi's file, always with somebody else's provider in it beside whatever is asked."""
    listed = {"stranger": copy.deepcopy(STRANGER)}
    listed.update(copy.deepcopy(providers or {}))

    return {"providers": listed}


class APiThatHasNeverRunHasSomethingToBePointedAt(unittest.TestCase):
    """It writes its model list on first run. Where it has been installed and not yet
    started there is nothing to add a provider to, and this is the shape it documents."""

    def test_the_empty_list_is_one_this_can_write_into(self):
        given = pointed(empty(), HERE, "llamacpp-cuda", MODELS)

        self.assertEqual(["llamacpp-cuda"], list(given.document["providers"]))

    def test_it_holds_no_provider_of_its_own(self):
        self.assertEqual({"providers": {}}, empty())


class OnlyThisRoutersProviderIsTouched(unittest.TestCase):
    def test_another_provider_comes_back_unchanged(self):
        given = config()

        written = pointed(given, HERE, "llamacpp-cuda", MODELS).document

        self.assertEqual(STRANGER, written["providers"]["stranger"])

    def test_the_document_handed_in_is_not_the_one_changed(self):
        """It is read from a file and may be used again; changing it in place would
        make the preview and the write disagree."""
        given = config()

        pointed(given, HERE, "llamacpp-cuda", MODELS)

        self.assertEqual(config(), given)

    def test_keys_of_this_provider_that_are_not_ours_stay(self):
        mine = {"api": "openai-completions", "baseUrl": HERE.base_url, "models": [],
                "compat": {"maxTokensField": "max_completion_tokens"},
                "headers": {"x-note": "set by hand"}}

        written = pointed(config({"llamacpp-cuda": mine}), HERE, "other", MODELS)
        provider = written.document["providers"]["llamacpp-cuda"]

        self.assertEqual({"maxTokensField": "max_completion_tokens"}, provider["compat"])
        self.assertEqual({"x-note": "set by hand"}, provider["headers"])

    def test_a_file_shaped_some_other_way_is_refused(self):
        with self.assertRaises(UnknownShape):
            pointed({"models": []}, HERE, "llamacpp-cuda", MODELS)


class WhichProviderThisRouterIs(unittest.TestCase):
    def test_the_one_already_pointing_at_it_whatever_it_is_called(self):
        """A provider renamed in pi is still this router."""
        renamed = {"baseUrl": HERE.base_url, "models": []}

        written = pointed(config({"my-box": renamed}), HERE, "llamacpp-cuda", MODELS)

        self.assertEqual({"stranger", "my-box"}, set(written.document["providers"]))

    def test_a_provider_at_another_port_is_a_different_router(self):
        other = {"baseUrl": "http://gpu-box:18082/v1", "models": []}

        written = pointed(config({"distributed": other}), HERE, "llamacpp-cuda", MODELS)

        self.assertEqual({"stranger", "distributed", "llamacpp-cuda"},
                         set(written.document["providers"]))

    def test_a_port_this_one_is_the_start_of_is_a_different_router(self):
        """:18081 and :180810 are two machines, however alike they read."""
        longer = {"baseUrl": "http://gpu-box:180810/v1", "models": []}

        written = pointed(config({"other": longer}), HERE, "llamacpp-cuda", MODELS)
        providers = written.document["providers"]

        self.assertIn("llamacpp-cuda", providers)
        self.assertEqual([], providers["other"]["models"])

    def test_the_named_one_is_repointed_rather_than_duplicated(self):
        """The address changed; leaving the old entry behind leaves pi two of them."""
        moved = {"baseUrl": "http://old-box:18081/v1", "models": []}

        written = pointed(config({"llamacpp-cuda": moved}), HERE, "llamacpp-cuda",
                          MODELS)
        providers = written.document["providers"]

        self.assertEqual({"stranger", "llamacpp-cuda"}, set(providers))
        self.assertEqual(HERE.base_url, providers["llamacpp-cuda"]["baseUrl"])

    def test_a_machine_that_has_never_been_pointed_here(self):
        written = pointed(config(), HERE, "llamacpp-cuda", MODELS)
        made = written.document["providers"]["llamacpp-cuda"]

        self.assertEqual(HERE.base_url, made["baseUrl"])
        self.assertEqual("openai-completions", made["api"])
        self.assertTrue(made["apiKey"])
        self.assertEqual("max_tokens", made["compat"]["maxTokensField"])


class TheModelListIsWrittenNotMerged(unittest.TestCase):
    def test_the_list_is_what_the_router_serves_and_nothing_else(self):
        stale = {"baseUrl": HERE.base_url,
                 "models": [{"id": "a-model-long-gone", "contextWindow": 4096}]}

        written = pointed(config({"llamacpp-cuda": stale}), HERE, "llamacpp-cuda",
                          MODELS)
        listed = written.document["providers"]["llamacpp-cuda"]["models"]

        self.assertEqual([one.id for one in MODELS], [one["id"] for one in listed])

    def test_every_field_pi_reads_is_there(self):
        written = pointed(config(), HERE, "llamacpp-cuda", MODELS)
        first = written.document["providers"]["llamacpp-cuda"]["models"][0]

        self.assertEqual({"id": "qwen3.8-52k-q4-mtp",
                          "name": "qwen3.8-52k-q4-mtp for people",
                          "contextWindow": 52000,
                          "maxTokens": 16384,
                          "input": ["text"],
                          "reasoning": True},
                         first)


class PiMustCompactLaterThanThePolicyDoes(unittest.TestCase):
    """An extension cannot switch pi's compaction off, only get there first."""

    def test_a_settings_file_that_says_nothing_about_compaction(self):
        written = compacted({"theme": "dark"}, ROOM)

        self.assertEqual({"enabled": True, "reserveTokens": 8000,
                          "keepRecentTokens": 4000},
                         written.document["compaction"])
        self.assertEqual("dark", written.document["theme"])

    def test_numbers_that_already_work_are_left_alone(self):
        held = {"compaction": {"enabled": True, "reserveTokens": 8192,
                               "keepRecentTokens": 4000}}

        written = compacted(held, ROOM)

        self.assertEqual((), written.changes)
        self.assertEqual(held, written.document)

    def test_a_reserve_the_policy_would_stand_aside_for_is_brought_under_it(self):
        held = {"compaction": {"enabled": True, "reserveTokens": 16384,
                               "keepRecentTokens": 4000}}

        written = compacted(held, ROOM)

        self.assertEqual(8000, written.document["compaction"]["reserveTokens"])
        self.assertEqual(1, len(written.changes))

    def test_the_largest_reserve_still_behind_the_policy_stands(self):
        """The extension flags at 'at or above', so 8399 is behind it and 8400 is not."""
        behind = compacted({"compaction": {"enabled": True, "reserveTokens": 8399,
                                           "keepRecentTokens": 4000}}, ROOM)
        level = compacted({"compaction": {"enabled": True, "reserveTokens": 8400,
                                          "keepRecentTokens": 4000}}, ROOM)

        self.assertEqual(8399, behind.document["compaction"]["reserveTokens"])
        self.assertEqual(8000, level.document["compaction"]["reserveTokens"])

    def test_keeping_more_than_the_policy_keeps_is_brought_down(self):
        """pi answers 'nothing to compact' before the policy is ever asked where to cut."""
        held = {"compaction": {"enabled": True, "reserveTokens": 8000,
                               "keepRecentTokens": 20000}}

        written = compacted(held, ROOM)

        self.assertEqual(4000, written.document["compaction"]["keepRecentTokens"])

    def test_keeping_exactly_what_the_policy_keeps_is_allowed(self):
        held = {"compaction": {"enabled": True, "reserveTokens": 8000,
                               "keepRecentTokens": 6860}}

        written = compacted(held, ROOM)

        self.assertEqual((), written.changes)

    def test_compaction_turned_off_is_turned_on(self):
        """The numbers mean nothing while pi is not compacting at all -- and turning it
        off also turns off its recovery from an overflow, which is the net under this."""
        written = compacted({"compaction": {"reserveTokens": 8000,
                                            "keepRecentTokens": 4000}}, ROOM)

        self.assertIs(True, written.document["compaction"]["enabled"])

    def test_a_number_pi_wrote_as_something_other_than_a_number(self):
        written = compacted({"compaction": {"reserveTokens": None,
                                            "keepRecentTokens": "half"}}, ROOM)

        self.assertEqual(8000, written.document["compaction"]["reserveTokens"])
        self.assertEqual(4000, written.document["compaction"]["keepRecentTokens"])


if __name__ == "__main__":
    unittest.main()
