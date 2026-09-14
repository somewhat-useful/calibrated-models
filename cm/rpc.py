"""A slave: a machine lending its card to the router's machine through a llama.cpp worker.

Nothing here is opened or run. This says what the worker is, where it is reached, how an
address a person typed names one, and what the worker is started with.
"""

from dataclasses import dataclass

from .place import Endpoint
from .units import Port

# What llama.cpp's worker is called, and where it listens unless told otherwise.
WORKER = "ggml-rpc-server.exe"
DEFAULT_PORT = Port(50052)


@dataclass(frozen=True)
class Unreadable:
    """An address that names no worker, and the words saying why."""

    why: str


def endpoint(said: str, port: Port) -> Endpoint | Unreadable:
    """Where a worker is, out of an address as a person types it: a host, or a host and
    a port. A port in the address is the one meant: it was written more recently than
    the default. A pasted URL is reduced to its host and port."""
    named = said.strip().split("://")[-1].split("/")[0]
    if not named:
        return Unreadable(f"{said!r} names no host")

    host, colon, written_port = named.rpartition(":")
    if not colon:
        return Endpoint(named, port)
    if not host or not written_port.isdigit() or not 1 <= int(written_port) <= 65535:
        return Unreadable(f"{said!r} is not a host, or a host and a port")

    return Endpoint(host, Port(int(written_port)))


def written(endpoint: Endpoint) -> str:
    """A worker's address the way llama.cpp and the settings file both take it."""
    return f"{endpoint.host}:{endpoint.port}"


def arguments(port: Port, device: str) -> tuple[str, ...]:
    """The worker's command line.

    Every address, because it exists to be called from another machine; the firewall
    rule is what keeps that to the local subnet. One card and nothing else: left to
    itself the worker offers the processor too, and the router would then be free to put
    layers in this machine's memory over the network, the slowest placement there is.
    The cache keeps the tensors it is sent on this machine's disk, so loading the same
    model again does not send them again.
    """
    return ("--host", "0.0.0.0", "--port", str(port), "--device", device, "--cache")
