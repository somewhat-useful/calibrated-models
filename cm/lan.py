"""The ports this program listens on, as the rest of the network sees them.

Windows turns away an inbound connection to a newly bound port, so a router listening on
every address answers this machine and nothing else until a rule exists. So does a
slave's worker, whose whole purpose is to be called from another machine. This says what
those rules are.

The local subnet, and not any address: nothing stands in front of either, and whoever
reaches the router can load models, spend the card and read whatever a conversation
carries, while whoever reaches a worker can run anything on its card. Widening that is a
decision somebody takes deliberately, by writing a rule of their own, and not one this
takes on their behalf.

A port in, what to ask of netsh out. Nothing here is opened or run.
"""

from .units import Port

# netsh's own word for the network this machine is on.
SUBNET = "localsubnet"

# Private, not public. Where Windows has classified the network as public, this endpoint
# has no business answering whoever else is on it -- and the rule not applying there is
# the right answer rather than a fault.
PROFILE = "private"

DESCRIPTION = "Allow the llama.cpp router to be called from the local subnet."

WORKER_DESCRIPTION = ("Allow a llama.cpp router on the local subnet to use this machine's "
                      "card.")


def rule(port: Port) -> str:
    """What the rule is called.

    The port is part of the name. A router moved to another port needs another rule, and
    the one it had should not be left behind admitting a port nothing serves.
    """
    return f"llama.cpp router ({port})"


def worker_rule(port: Port) -> str:
    """What the worker's rule is called, the port in it for the same reason."""
    return f"llama.cpp RPC worker ({port})"


def opened(port: Port) -> tuple[str, ...]:
    """netsh, asked to admit the local subnet to the router's port."""
    return _opened(rule(port), port, DESCRIPTION)


def worker_opened(port: Port) -> tuple[str, ...]:
    """netsh, asked to admit the local subnet to the worker's port."""
    return _opened(worker_rule(port), port, WORKER_DESCRIPTION)


def closed(port: Port) -> tuple[str, ...]:
    """netsh, asked to take the router's rule away."""
    return _closed(rule(port))


def worker_closed(port: Port) -> tuple[str, ...]:
    """netsh, asked to take the worker's rule away."""
    return _closed(worker_rule(port))


def shown(port: Port) -> tuple[str, ...]:
    """netsh, asked whether the router's rule is there. It answers by exit code."""
    return _shown(rule(port))


def worker_shown(port: Port) -> tuple[str, ...]:
    """netsh, asked whether the worker's rule is there. It answers by exit code."""
    return _shown(worker_rule(port))


def _opened(name: str, port: Port, description: str) -> tuple[str, ...]:
    return ("netsh", "advfirewall", "firewall", "add", "rule",
            f"name={name}",
            "dir=in",
            "action=allow",
            "protocol=TCP",
            f"localport={port}",
            f"profile={PROFILE}",
            f"remoteip={SUBNET}",
            f"description={description}")


def _closed(name: str) -> tuple[str, ...]:
    return ("netsh", "advfirewall", "firewall", "delete", "rule", f"name={name}")


def _shown(name: str) -> tuple[str, ...]:
    return ("netsh", "advfirewall", "firewall", "show", "rule", f"name={name}")
