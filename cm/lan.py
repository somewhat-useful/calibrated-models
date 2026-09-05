"""The router's port, as the rest of the network sees it.

Windows turns away an inbound connection to a newly bound port, so a router listening on
every address answers this machine and nothing else until a rule exists. This says what
that rule is.

The local subnet, and not any address: nothing stands in front of the endpoint, and
whoever reaches it can load models, spend the card and read whatever a conversation
carries. Widening that is a decision somebody takes deliberately, by writing a rule of
their own, and not one this takes on their behalf.

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


def rule(port: Port) -> str:
    """What the rule is called.

    The port is part of the name. A router moved to another port needs another rule, and
    the one it had should not be left behind admitting a port nothing serves.
    """
    return f"llama.cpp router ({port})"


def opened(port: Port) -> tuple[str, ...]:
    """netsh, asked to admit the local subnet to that port."""
    return ("netsh", "advfirewall", "firewall", "add", "rule",
            f"name={rule(port)}",
            "dir=in",
            "action=allow",
            "protocol=TCP",
            f"localport={port}",
            f"profile={PROFILE}",
            f"remoteip={SUBNET}",
            f"description={DESCRIPTION}")


def closed(port: Port) -> tuple[str, ...]:
    """netsh, asked to take that rule away."""
    return ("netsh", "advfirewall", "firewall", "delete", "rule", f"name={rule(port)}")


def shown(port: Port) -> tuple[str, ...]:
    """netsh, asked whether the rule is there. It answers by exit code."""
    return ("netsh", "advfirewall", "firewall", "show", "rule", f"name={rule(port)}")
