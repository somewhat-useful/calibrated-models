"""Invariants of reading back what a profile across several devices holds on one of them.

`vram` watches the card a desktop is drawn on. A profile says what it holds on every
device it uses, and what is read is the figure for that card: nothing where the profile
does not use it, and the one figure of a profile written for a machine's only card.
"""

import unittest

from cm.catalog import parse
from cm.place import Pipeline
from cm.render import preset
from cm.units import Mib
from test_render import MACHINE, MEMORY, config, placed
from test_render_devices import FAST, SLAVE, SLOW, across

HELD = "held from the moment this profile loads"

PRESET = f"""version = 1

[*]
threads = 16

[across-both]
; VRAM REQUIRED: 7000 MiB on CUDA1, 14000 MiB on CUDA0, {HELD}
ctx-size = 150000

[on-the-fast-one]
; VRAM REQUIRED: 10000 MiB on CUDA0, {HELD}
ctx-size = 262000

[with-the-slave]
; VRAM REQUIRED: 5000 MiB on RPC0, 7100 MiB on CUDA1, 14200 MiB on CUDA0, {HELD}
ctx-size = 262000
"""

ONLY_CARD = """version = 1

[alone]
; VRAM REQUIRED: 15285 MiB of video memory, held from the moment this profile loads
ctx-size = 109000
"""


class WhatIsReadIsTheFigureForTheCardWatched(unittest.TestCase):
    def test_each_profile_comes_back_with_what_it_holds_on_that_card(self):
        self.assertEqual([Mib(14000), Mib(10000), Mib(14200)],
                         [one.needs for one in parse(PRESET, "CUDA0")])

    def test_a_profile_that_does_not_use_the_card_holds_nothing_on_it(self):
        self.assertEqual([Mib(7000), Mib(0), Mib(7100)],
                         [one.needs for one in parse(PRESET, "CUDA1")])

    def test_a_requirement_naming_no_device_is_the_only_cards(self):
        self.assertEqual([Mib(15285)], [one.needs for one in parse(ONLY_CARD, "CUDA0")])


class WhatCalibrateWritesIsWhatVramReadsOnEveryCard(unittest.TestCase):
    def test_the_figure_read_for_a_card_is_what_the_placement_holds_there(self):
        settings = across((SLAVE, SLOW, FAST), (24, 15, 27), (2048, 1024, 2048),
                          pipeline=Pipeline.OFF)
        written = preset(config(), MACHINE, MEMORY, (placed("model", settings),))

        for device, held in (("RPC0", 12288 - 2048), ("CUDA1", 8192 - 1024),
                             ("CUDA0", 16303 - 2048)):
            with self.subTest(device=device):
                self.assertEqual([Mib(held)], [one.needs for one in parse(written, device)])


if __name__ == "__main__":
    unittest.main()
