"""Invariants of the command line the router is started with.

What matters here is what is *not* on it as much as what is. Every window, every cache
and every placement lives in the preset, which is also what a client reads back over
HTTP to learn what it is talking to; a flag written here instead would serve a model
under a configuration nothing else knows about.
"""

import unittest
from pathlib import Path

from cm.advise import Pid
from cm.serving import (DEFAULT_HOST, DEFAULT_IDLE, DEFAULT_PORT,
                        DEFAULT_RESIDENT, Occupied, Serving, arguments,
                        in_the_way, logs, still_held, url, written)
from cm.units import Port, Seconds
from places import somewhere

PRESET = somewhere("calibrated-models", "llamacpp.models.ini")
LOGS = somewhere("llama.cpp", "logs")


def serving(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
            resident: int = DEFAULT_RESIDENT, idle: int = DEFAULT_IDLE) -> Serving:
    return Serving(host=host, port=Port(port), resident=resident, idle=Seconds(idle),
                   logs=LOGS)


class TheRouterIsToldWhereEverythingIs(unittest.TestCase):
    def line(self, **rest) -> tuple[str, ...]:
        return arguments(PRESET, serving(**rest), LOGS / "router.log")

    def test_the_preset_it_serves_from(self):
        given = self.line()

        self.assertEqual(str(PRESET), given[given.index("--models-preset") + 1])

    def test_where_it_listens(self):
        given = self.line(host="127.0.0.1", port=18099)

        self.assertEqual("127.0.0.1", given[given.index("--host") + 1])
        self.assertEqual("18099", given[given.index("--port") + 1])

    def test_what_it_keeps_loaded_and_for_how_long(self):
        given = self.line(resident=2, idle=60)

        self.assertEqual("2", given[given.index("--models-max") + 1])
        self.assertEqual("60", given[given.index("--sleep-idle-seconds") + 1])

    def test_where_it_writes(self):
        given = self.line()

        self.assertEqual(str(LOGS / "router.log"),
                         given[given.index("--log-file") + 1])

    def test_the_numbers_are_written_as_a_command_line_carries_them(self):
        """Text, in the invariant form. A locale that groups thousands would offer
        llama.cpp a port it cannot parse."""
        for one in self.line(port=18099, idle=900):
            with self.subTest(one=one):
                self.assertNotIn(",", one)


class WhatIsInThePresetIsNotOnTheCommandLine(unittest.TestCase):
    ARGUMENTS = arguments(PRESET, serving(), LOGS / "router.log")

    def test_no_model_is_named(self):
        """A model named here would be loaded at start-up and held: the router is
        started empty and loads what a request asks for."""
        self.assertNotIn("--model", self.ARGUMENTS)

    def test_no_placement_is_named(self):
        for flag in ("--ctx-size", "--gpu-layers", "--n-cpu-moe", "--cache-type-k",
                     "--cache-type-v", "--threads", "--cache-ram"):
            with self.subTest(flag=flag):
                self.assertNotIn(flag, self.ARGUMENTS)

    def test_the_built_in_interface_is_off(self):
        """Nothing serves it a page, and it answers on the same port as the API."""
        self.assertIn("--no-ui", self.ARGUMENTS)


class TheCommandIsPrintedAsAPersonWouldHaveToTypeIt(unittest.TestCase):
    def test_a_path_with_a_space_is_quoted(self):
        """Where a release was unpacked is where somebody put it, and half the
        directories on a Windows machine have a space in them."""
        spaced = somewhere("llama cpp", "llama-server.exe")

        self.assertEqual(f'"{spaced}"', written(spaced, ()))

    def test_a_path_without_one_is_left_alone(self):
        self.assertEqual("llama-server.exe", written(Path("llama-server.exe"), ()))

    def test_the_server_comes_first_and_the_arguments_follow(self):
        given = written(Path("llama-server.exe"), ("--port", "18081"))

        self.assertEqual("llama-server.exe --port 18081", given)


class ARunningRouterLeavesFourFilesBehind(unittest.TestCase):
    WRITTEN = logs(LOGS)

    def test_they_are_all_in_the_log_directory(self):
        for one in (self.WRITTEN.router, self.WRITTEN.out, self.WRITTEN.err,
                    self.WRITTEN.pid):
            with self.subTest(one=one):
                self.assertEqual(LOGS, one.parent)

    def test_no_two_of_them_are_the_same_file(self):
        """The two standard streams cannot share a file: they are opened separately and
        each writes from its own offset."""
        written_to = (self.WRITTEN.router, self.WRITTEN.out, self.WRITTEN.err,
                      self.WRITTEN.pid)

        self.assertEqual(len(written_to), len(set(written_to)))


class WhereClientsArePointed(unittest.TestCase):
    def test_the_address_it_binds_is_the_address_it_is_reported_on(self):
        self.assertEqual("http://0.0.0.0:18081/v1", url(serving()))

    def test_a_router_told_to_listen_on_one_address_only(self):
        self.assertEqual("http://127.0.0.1:18099/v1",
                         url(serving(host="127.0.0.1", port=18099)))


OURS = Pid(1000)


def occupied(running: tuple[int, ...] = (),
             on_the_port: tuple[int, ...] = ()) -> Occupied:
    return Occupied(running=frozenset(Pid(one) for one in running),
                    on_the_port=frozenset(Pid(one) for one in on_the_port))


class OnlyTheServerHoldingThisPortIsStopped(unittest.TestCase):
    """Both tests have to pass, and neither on its own is a reason to end anything.

    Running the server is not one: another copy of these scripts is serving somebody on
    a port of its own, and a worker the server started runs the same executable and is
    its child. Holding the port is not one either: something else there is somebody's
    program, and ending it is a person's decision rather than a start's.
    """

    def test_a_clear_machine_asks_nothing_to_go(self):
        self.assertEqual((), in_the_way(occupied(), OURS))

    def test_the_server_on_this_port_is_what_goes(self):
        given = occupied(running=(7,), on_the_port=(7,))

        self.assertEqual((Pid(7),), in_the_way(given, OURS))

    def test_a_server_on_some_other_port_is_left_alone(self):
        self.assertEqual((), in_the_way(occupied(running=(7,)), OURS))

    def test_something_on_this_port_that_is_not_the_server_is_left_alone(self):
        self.assertEqual((), in_the_way(occupied(on_the_port=(7,)), OURS))

    def test_two_processes_answering_separately_are_not_one_that_answers_both(self):
        """A server here and a stranger on the port is not a server on the port, and
        reading the two readings as one would end both of them."""
        given = occupied(running=(7,), on_the_port=(9,))

        self.assertEqual((), in_the_way(given, OURS))

    def test_this_process_is_never_asked_to_go(self):
        """On a foreground start it is the router being started, and on any other it is
        what was told to do the clearing. Ending it leaves the work half done."""
        given = occupied(running=(int(OURS),), on_the_port=(int(OURS),))

        self.assertEqual((), in_the_way(given, OURS))

    def test_they_come_back_in_a_settled_order(self):
        """Sets read off a machine arrive in whatever order Windows walked them, and a
        report that shuffles between runs is one nobody can compare."""
        given = occupied(running=(30, 10, 20), on_the_port=(20, 10, 30))

        self.assertEqual((Pid(10), Pid(20), Pid(30)), in_the_way(given, OURS))


class WhatMattersAfterwardsIsWhetherThePortIsFree(unittest.TestCase):
    """Not whether the processes aimed at are gone. A start has to bind, and a port
    held by something nobody tried to end stops it exactly as hard.
    """

    def test_a_free_port_is_free(self):
        self.assertEqual((), still_held(frozenset(), OURS))

    def test_whatever_has_the_port_is_named_whether_or_not_it_was_aimed_at(self):
        self.assertEqual((Pid(7),), still_held(frozenset({Pid(7)}), OURS))

    def test_our_own_listening_socket_does_not_count_against_us(self):
        """A foreground start is the router, and by the time this is asked again it is
        the one holding the port."""
        self.assertEqual((), still_held(frozenset({OURS}), OURS))


if __name__ == "__main__":
    unittest.main()
