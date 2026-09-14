"""The scheduled task that serves at boot: what it runs, and how it is registered.

The router is the one thing worth registering. It holds no model until a request names
one and unloads what it holds after an idle period, so a task that is always there costs
nothing while the machine is doing something else.

What it starts is this program rather than the server itself: the newest release is
resolved at every start, so installing one takes effect at the next boot with nothing
further to do. And it starts it in the foreground, which keeps the task alive for as
long as the server runs -- that is what makes the scheduler's restart-on-failure setting
mean anything, since a task that has already finished is not restarted.

Facts in, one document out. Nothing here is opened, run or registered.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import NewType
from xml.sax.saxutils import escape

from .units import Port

# What the task is called. router.py ends it before starting a router of its own -- one
# card takes one server -- so the two have to mean the same task, and they do by reading
# the name from here.
NAME = "llama.cpp router"

DESCRIPTION = ("llama.cpp router: serves the models the settings file names on one "
               "OpenAI-compatible endpoint, loading one on demand and unloading it "
               "when idle.")

# What the worker's task is called. slave.py ends it before starting a worker of its
# own -- one port takes one worker -- so the two read the name from here.
WORKER_NAME = "llama.cpp RPC worker"

WORKER_DESCRIPTION = ("llama.cpp RPC worker: lends this machine's card to a llama.cpp "
                      "router on another machine of the local network.")

# DOMAIN\\user, which is what the scheduler resolves to an account.
Account = NewType("Account", str)


@dataclass(frozen=True)
class Runs:
    """What the task starts at boot, and where it starts it."""

    command: Path
    arguments: str
    working: Path


def arguments(settings: Path) -> str:
    """The router, told to hold the console the task gives it.

    The settings file is named in full rather than left to the working directory: a task
    is started by the scheduler, and where a service looks for a file should not depend
    on what a directory happened to hold.
    """
    return f'-m cm.router start --foreground --settings "{settings}"'


def worker_arguments(port: Port, settings: Path) -> str:
    """The worker, told to hold the console the task gives it, on the port the
    router's machine was given, and the settings file named in full for the same
    reason as the router's."""
    return f'-m cm.slave start --foreground --port {port} --settings "{settings}"'


def document(started: Runs, account: Account) -> str:
    """The task as the scheduler reads it.

    S4U is why no password appears here or anywhere else: the task runs as this account
    without one being stored, which is all a service needs that reads local files and
    binds a local port. Nothing here asks for elevation either -- a server reading model
    files and holding a port has no use for it, and a task that runs elevated is one
    more thing on the machine that does.
    """
    return _document(started, account, DESCRIPTION)


def worker_document(started: Runs, account: Account) -> str:
    """The worker's task: the router's in everything -- this account without a
    password, at boot, restarted if it fails -- but what it says it is.
    """
    return _document(started, account, WORKER_DESCRIPTION)


def _document(started: Runs, account: Account, description: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.3" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>{escape(description)}</Description>
  </RegistrationInfo>
  <Principals>
    <Principal id="Author">
      <UserId>{escape(account)}</UserId>
      <LogonType>S4U</LogonType>
    </Principal>
  </Principals>
  <Settings>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <RestartOnFailure>
      <Count>3</Count>
      <Interval>PT1M</Interval>
    </RestartOnFailure>
    <StartWhenAvailable>true</StartWhenAvailable>
  </Settings>
  <Triggers>
    <BootTrigger />
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(str(started.command))}</Command>
      <Arguments>{escape(started.arguments)}</Arguments>
      <WorkingDirectory>{escape(str(started.working))}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""
