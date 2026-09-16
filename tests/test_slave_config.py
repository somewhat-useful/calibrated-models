"""Invariants of the settings a machine with several cards and a slave reads.

The slave is the one thing in the file nothing on this machine can check: its card is on
another machine. So what the file says about it is taken as said, and every way of
saying it wrongly is refused with the line saying what to fix.
"""

import unittest

from cm.config import ConfigError, NoSlave, gibibytes, slaved, unslaved
from cm.place import (DEFAULT_MULTI_GPU_RESERVE, DEFAULT_NO_MONITOR_RESERVE,
                      DEFAULT_SLAVE_RESERVE, Endpoint, Worker)
from cm.units import Mib, Port
from test_config import BARE, FILE, MODEL_ROOT, parse, refused

WORKER = Worker(endpoint=Endpoint("worker", Port(50060)), memory=Mib(12288),
                reserve=Mib(3072))


def with_slave(*lines):
    return BARE + "\n[slave]\n" + "\n".join(lines) + "\n"


class TheMultiGpuReserveIsItsOwnSetting(unittest.TestCase):
    def test_absent_it_is_the_default(self):
        self.assertEqual(DEFAULT_MULTI_GPU_RESERVE, parse(BARE).reserve_multi_gpu)

    def test_written_it_is_what_was_written(self):
        self.assertEqual(Mib(3072),
                         parse("reserve_multi_gpu_mib = 3072\n" + BARE).reserve_multi_gpu)

    def test_it_does_not_move_the_one_card_reserve(self):
        read = parse("reserve_multi_gpu_mib = 3072\n" + BARE)

        self.assertNotEqual(read.reserve, read.reserve_multi_gpu)

    def test_a_reserve_that_is_not_a_number_is_refused(self):
        self.assertEqual("reserve_multi_gpu_mib must be a whole number",
                         refused("reserve_multi_gpu_mib = 'lots'\n" + BARE))


class TheReserveForACardWithoutAMonitorIsItsOwnSetting(unittest.TestCase):
    def test_absent_it_is_the_default(self):
        self.assertEqual(DEFAULT_NO_MONITOR_RESERVE, parse(BARE).reserve_no_monitor)

    def test_written_it_is_what_was_written(self):
        self.assertEqual(
            Mib(256), parse("reserve_no_monitor_mib = 256\n" + BARE).reserve_no_monitor)

    def test_it_moves_neither_of_the_other_two(self):
        read = parse("reserve_no_monitor_mib = 256\n" + BARE)

        self.assertNotEqual(read.reserve, read.reserve_no_monitor)
        self.assertNotEqual(read.reserve_multi_gpu, read.reserve_no_monitor)

    def test_a_reserve_that_is_not_a_number_is_refused(self):
        self.assertEqual("reserve_no_monitor_mib must be a whole number",
                         refused("reserve_no_monitor_mib = 'lots'\n" + BARE))


class ASlaveIsReadAsWritten(unittest.TestCase):
    def test_a_file_naming_none_has_none(self):
        self.assertEqual(NoSlave(), parse(BARE).slave)

    def test_address_memory_and_reserve_are_read(self):
        self.assertEqual(WORKER, parse(with_slave("address = 'worker:50060'",
                                                  "memory = '12G'",
                                                  "reserve_mib = 3072")).slave)

    def test_an_address_without_a_port_is_the_workers_own(self):
        read = parse(with_slave("address = 'worker'", "memory = 12")).slave

        self.assertEqual(Endpoint("worker", Port(50052)), read.endpoint)

    def test_the_reserve_left_unwritten_is_the_default(self):
        read = parse(with_slave("address = 'worker'", "memory = 12")).slave

        self.assertEqual(DEFAULT_SLAVE_RESERVE, read.reserve)

    def test_memory_is_in_gibibytes_however_it_is_written(self):
        for written in ("12", "'12'", "'12G'", "'12Gb'", "'12GiB'"):
            with self.subTest(written=written):
                read = parse(with_slave("address = 'worker'", f"memory = {written}"))
                self.assertEqual(Mib(12288), read.slave.memory)


class ASlaveWrittenWronglyIsRefused(unittest.TestCase):
    MEMORY = "slave: memory is the size of its card in gibibytes: 12, '12G' or '12GiB'"

    def test_one_that_is_not_a_table(self):
        self.assertEqual("slave must be a table: [slave], with address = and memory = "
                         "under it", refused(MODEL_ROOT + "\nslave = 'worker'\n"))

    def test_one_with_no_address(self):
        self.assertEqual("slave: address is not set", refused(with_slave("memory = 12")))

    def test_an_address_that_is_not_a_host_and_port(self):
        said = refused(with_slave("address = 'worker:port'", "memory = 12"))

        self.assertTrue(said.startswith("slave: address"), said)

    def test_memory_that_is_not_a_size(self):
        for written in ("'50%'", "0", "'plenty'", "true"):
            with self.subTest(written=written):
                self.assertEqual(self.MEMORY,
                                 refused(with_slave("address = 'worker'",
                                                    f"memory = {written}")))

    def test_memory_left_out(self):
        self.assertEqual(self.MEMORY, refused(with_slave("address = 'worker'")))

    def test_a_reserve_that_is_not_a_number(self):
        self.assertEqual("slave: reserve_mib must be a whole number",
                         refused(with_slave("address = 'worker'", "memory = 12",
                                            "reserve_mib = 'lots'")))

    def test_memory_typed_on_a_command_line_is_read_the_same_way(self):
        self.assertEqual(Mib(8192), gibibytes("8G"))
        with self.assertRaises(ConfigError):
            gibibytes("8%")


class TheSlaveIsWrittenIntoTheFileAndTakenOut(unittest.TestCase):
    def test_a_file_naming_none_names_the_one_written(self):
        self.assertEqual(WORKER, parse(slaved(BARE, WORKER)).slave)

    def test_everything_already_in_the_file_stays(self):
        written = slaved(BARE, WORKER)

        self.assertTrue(written.startswith(BARE.rstrip("\n")))
        self.assertEqual(parse(BARE).models, parse(written).models)

    def test_writing_it_twice_is_writing_it_once(self):
        once = slaved(BARE, WORKER)

        self.assertEqual(once, slaved(once, WORKER))
        self.assertEqual(1, once.count("[slave]"))

    def test_another_slave_replaces_the_one_there(self):
        other = Worker(endpoint=Endpoint("elsewhere", Port(50052)), memory=Mib(8192),
                       reserve=Mib(2048))

        written = slaved(slaved(BARE, WORKER), other)

        self.assertEqual(other, parse(written).slave)
        self.assertEqual(1, written.count("[slave]"))

    def test_taking_it_out_leaves_a_file_naming_none(self):
        self.assertEqual(NoSlave(), parse(unslaved(slaved(BARE, WORKER))).slave)

    def test_what_follows_a_slave_in_the_middle_of_the_file_stays(self):
        text = (f"{MODEL_ROOT}\n\n[slave]\naddress = 'worker'\nmemory = 8\n\n"
                f"# the model\n[models.'qwen3.8']\n{FILE}\n")

        taken = unslaved(text)

        self.assertIn("# the model", taken)
        self.assertEqual(NoSlave(), parse(taken).slave)
        self.assertEqual(parse(text).models, parse(taken).models)

    def test_a_file_naming_none_is_left_as_it_is(self):
        self.assertEqual(BARE, unslaved(BARE))

    def test_a_slave_written_inline_is_refused_rather_than_half_rewritten(self):
        inline = MODEL_ROOT + "\nslave = { address = 'worker', memory = 8 }\n"

        with self.assertRaises(ConfigError) as refusal:
            slaved(inline, WORKER)

        self.assertIn("by hand", str(refusal.exception))


if __name__ == "__main__":
    unittest.main()
