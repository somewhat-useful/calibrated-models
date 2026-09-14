"""Invariants of what a slave registers, admits and runs.

A slave is set up once and then left alone on a machine nobody is watching, so what it is
handed has to say exactly what was meant: the worker at boot under this account without
a password, its port open to the local subnet and nothing wider, one card offered and
nothing else.
"""

import unittest
from pathlib import Path
from xml.etree import ElementTree

from cm import config, lan, rpc, task
from cm.config import DEFAULT_CUDA, DEFAULT_KEPT
from cm.place import Endpoint
from cm.units import Port
from cm.upstream import Cuda
from places import somewhere
from test_lan import asked
from test_task import SCHEMA, WHO, one

PORT = Port(50052)
SETTINGS = somewhere("calibrated models", "settings.toml")
STARTED = task.Runs(command=somewhere("python", "python.exe"),
                    arguments=task.worker_arguments(PORT, SETTINGS),
                    working=somewhere("calibrated models"))


def read(document: str) -> ElementTree.Element:
    return ElementTree.fromstring(document)


class TheWorkerIsStartedTheWayTheRouterIs(unittest.TestCase):
    def test_it_runs_the_worker_command_in_the_foreground_on_its_port(self):
        arguments = one(read(task.worker_document(STARTED, WHO)),
                        "Actions/Exec/Arguments").text

        self.assertIn("-m cm.slave start --foreground", arguments)
        self.assertIn(f"--port {PORT}", arguments)
        self.assertIn(f'--settings "{SETTINGS}"', arguments)

    def test_at_boot_under_the_account_without_a_password_and_restarted(self):
        document = read(task.worker_document(STARTED, WHO))

        self.assertEqual("S4U", one(document, "Principals/Principal/LogonType").text)
        self.assertEqual(WHO, one(document, "Principals/Principal/UserId").text)
        one(document, "Triggers/BootTrigger")
        one(document, "Settings/RestartOnFailure")

    def test_it_says_it_is_the_worker_and_the_router_still_says_it_is_the_router(self):
        worker = one(read(task.worker_document(STARTED, WHO)),
                     "RegistrationInfo/Description").text
        router = one(read(task.document(STARTED, WHO)), "RegistrationInfo/Description").text

        self.assertEqual(task.WORKER_DESCRIPTION, worker)
        self.assertEqual(task.DESCRIPTION, router)

    def test_the_two_tasks_are_different_tasks(self):
        self.assertNotEqual(task.NAME, task.WORKER_NAME)


class TheWorkersPortIsOpenedToTheLocalSubnetOnly(unittest.TestCase):
    def test_its_rule_is_not_the_routers(self):
        self.assertNotEqual(lan.rule(PORT), lan.worker_rule(PORT))
        self.assertIn(str(PORT), lan.worker_rule(PORT))

    def test_the_rule_added_is_the_rule_asked_after_and_taken_away(self):
        name = asked(lan.worker_opened(PORT))["name"]

        self.assertEqual(lan.worker_rule(PORT), name)
        self.assertEqual(name, asked(lan.worker_shown(PORT))["name"])
        self.assertEqual(name, asked(lan.worker_closed(PORT))["name"])

    def test_inbound_tcp_on_its_port_from_the_local_subnet_on_the_private_profile(self):
        said = asked(lan.worker_opened(PORT))

        self.assertEqual((lan.SUBNET, lan.PROFILE, str(PORT), "TCP"),
                         (said["remoteip"], said["profile"], said["localport"],
                          said["protocol"]))
        self.assertIn("dir=in", lan.worker_opened(PORT))
        self.assertIn("action=allow", lan.worker_opened(PORT))


class AWorkersAddressIsReadTheWayAPersonTypesIt(unittest.TestCase):
    def test_a_host_alone_is_on_the_workers_own_port(self):
        self.assertEqual(Endpoint("worker", PORT), rpc.endpoint("worker", PORT))

    def test_a_port_written_in_the_address_is_the_one_meant(self):
        self.assertEqual(Endpoint("worker", Port(50060)),
                         rpc.endpoint("worker:50060", PORT))

    def test_a_pasted_url_is_reduced_to_host_and_port(self):
        self.assertEqual(Endpoint("worker", Port(50060)),
                         rpc.endpoint("tcp://worker:50060/", PORT))

    def test_what_names_no_worker_says_so(self):
        for said in ("", ":50052", "worker:port", "worker:70000", "worker:0"):
            with self.subTest(said=said):
                self.assertIsInstance(rpc.endpoint(said, PORT), rpc.Unreadable)

    def test_an_address_written_out_reads_back_as_itself(self):
        endpoint = Endpoint("worker", Port(50060))

        self.assertEqual(endpoint, rpc.endpoint(rpc.written(endpoint), PORT))


class TheWorkerOffersOneCardToEveryAddress(unittest.TestCase):
    def test_every_address_its_port_one_card_and_a_cache(self):
        line = rpc.arguments(PORT, "CUDA0")

        self.assertEqual("0.0.0.0", line[line.index("--host") + 1])
        self.assertEqual(str(PORT), line[line.index("--port") + 1])
        self.assertEqual("CUDA0", line[line.index("--device") + 1])
        self.assertIn("--cache", line)

    def test_its_streams_go_to_files_of_their_own(self):
        logs = rpc.logs(somewhere("logs"))

        self.assertEqual(3, len({logs.out, logs.err, logs.pid}))
        for path in (logs.out, logs.err, logs.pid):
            with self.subTest(path=path.name):
                self.assertEqual(somewhere("logs"), path.parent)


class ASlaveReadsOnlyWhatItNeeds(unittest.TestCase):
    def test_a_machine_with_no_settings_file_runs_on_the_defaults(self):
        self.assertEqual(config.Lending(cuda=DEFAULT_CUDA, keep_releases=DEFAULT_KEPT,
                                        logs=Path("logs")),
                         config.lending(""))

    def test_what_the_file_says_is_what_it_reads(self):
        read_back = config.lending("cuda_version = '12.4'\nkeep_releases = 3\n"
                                   "log_dir = 'worker-logs'\n")

        self.assertEqual(config.Lending(cuda=Cuda("12.4"), keep_releases=3,
                                        logs=Path("worker-logs")), read_back)

    def test_a_file_naming_no_models_and_no_library_is_not_refused(self):
        """What parse would refuse for want of a model root is nothing a slave needs."""
        self.assertEqual(DEFAULT_CUDA, config.lending("reserve_mib = 1024\n").cuda)


if __name__ == "__main__":
    unittest.main()
