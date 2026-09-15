"""What this logged-in session is running, and what closes a program.

Four readings, none of which decides anything:

    the performance counters   how much video memory each process holds
    a process snapshot         what each process is called and whose child it is
    the windows on screen      which processes have one up
    the executable itself      where it lives and what it says it belongs to, which is
                               how Windows' own is told from the person's

Video memory per process does not come from nvidia-smi here. Under WDDM the driver does
not attribute it, and the number that exists is the one Windows keeps in its own
counters. It is not free of the same trouble -- the compositor is credited with surfaces
belonging to the applications drawing through it -- but that is dealt with where the
crediting is decided, not here.
"""

# Annotations are left unevaluated: they name ctypes.WinDLL, which exists only on
# Windows, and this module is imported on every system even where none of it is called.
from __future__ import annotations

import ctypes
import os
import re
import sys
import tempfile
from collections.abc import Sequence
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path, PurePath

from .advise import Pid
from .desktop import Running
from .machine import UnreadableDevice
from .units import Mib, Port

_COUNTER = r"\GPU Process Memory(*)\Dedicated Usage"
_INSTANCE = re.compile(r"pid_(\d+)_")

_PDH_FMT_LARGE = 0x00000400
_PDH_MORE_DATA = 0x800007D2

_TH32CS_SNAPPROCESS = 0x00000002
_QUERY_LIMITED = 0x00001000
_TERMINATE = 0x00000001
_PATH = 32768
_GWL_EXSTYLE = -20
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_APPWINDOW = 0x00040000
_GW_OWNER = 4
_DWMWA_CLOAKED = 14
_WM_CLOSE = 0x0010

# EXTENDED_NAME_FORMAT.NameSamCompatible: the DOMAIN\user form.
_SAM_COMPATIBLE = 2

# SEE_MASK_NOCLOSEPROCESS, so that the run can be waited for; SW_HIDE, because the only
# window this should cost anybody is the prompt itself; ERROR_CANCELLED, which is how
# Windows reports a prompt answered no.
_NO_CLOSE_PROCESS = 0x00000040
_HIDDEN = 0
_CANCELLED = 1223
_FOREVER = 0xFFFFFFFF

# TCP_TABLE_CLASS.TCP_TABLE_OWNER_PID_LISTENER: the listening sockets and whose they
# are. Asked of iphlpapi rather than read off netstat, whose column headings and state
# words are in the language Windows was installed in -- a port reported free because a
# word did not match in English is a router that starts and cannot bind.
_LISTENERS = 3
_AF_INET = 2
_AF_INET6 = 23
_NO_ERROR = 0
_INSUFFICIENT_BUFFER = 122
_IPV6_ADDRESS = 16

_BYTES_PER_MIB = 1048576


class _CounterValue(ctypes.Structure):
    _fields_ = [("status", wintypes.DWORD), ("large", ctypes.c_longlong)]


class _CounterItem(ctypes.Structure):
    _fields_ = [("name", wintypes.LPWSTR), ("value", _CounterValue)]


class _ProcessEntry(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD),
                ("pid", wintypes.DWORD),
                ("heap", ctypes.POINTER(ctypes.c_ulong)),
                ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                ("parent", wintypes.DWORD), ("priority", ctypes.c_long),
                ("flags", wintypes.DWORD), ("exe", ctypes.c_wchar * 260)]


class _Elevated(ctypes.Structure):
    """SHELLEXECUTEINFOW, of which the verb, the command and the process are wanted."""

    _fields_ = [("size", wintypes.DWORD), ("mask", wintypes.ULONG),
                ("window", wintypes.HWND), ("verb", wintypes.LPCWSTR),
                ("file", wintypes.LPCWSTR), ("parameters", wintypes.LPCWSTR),
                ("directory", wintypes.LPCWSTR), ("show", ctypes.c_int),
                ("instance", wintypes.HINSTANCE), ("id_list", ctypes.c_void_p),
                ("class_name", wintypes.LPCWSTR), ("class_key", wintypes.HKEY),
                ("hot_key", wintypes.DWORD), ("icon", wintypes.HANDLE),
                ("process", wintypes.HANDLE)]


class _TcpListener(ctypes.Structure):
    """MIB_TCPROW_OWNER_PID, of which only two fields are wanted."""

    _fields_ = [("state", wintypes.DWORD), ("address", wintypes.DWORD),
                ("port", wintypes.DWORD), ("remote", wintypes.DWORD),
                ("remote_port", wintypes.DWORD), ("pid", wintypes.DWORD)]


class _Tcp6Listener(ctypes.Structure):
    """MIB_TCP6ROW_OWNER_PID. The same two fields, laid out differently."""

    _fields_ = [("address", ctypes.c_ubyte * _IPV6_ADDRESS),
                ("scope", wintypes.DWORD), ("port", wintypes.DWORD),
                ("remote", ctypes.c_ubyte * _IPV6_ADDRESS),
                ("remote_scope", wintypes.DWORD), ("remote_port", wintypes.DWORD),
                ("state", wintypes.DWORD), ("pid", wintypes.DWORD)]


def _enum_windows() -> type:
    """The callback EnumWindows is handed. Made when it is needed rather than at import:
    the calling convention it names exists only on Windows. ctypes hands back the same
    type every time it is asked."""
    return ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def _user32() -> ctypes.WinDLL:
    """user32 with its argument types declared.

    A window handle is a pointer. Left undeclared, ctypes passes it as a C int and the
    top half is lost -- which on this machine means a handle that names another window
    or none, silently.
    """
    library = ctypes.WinDLL("user32", use_last_error=True)

    library.EnumWindows.argtypes = [_enum_windows(), wintypes.LPARAM]
    library.IsWindowVisible.argtypes = [wintypes.HWND]
    library.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    library.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    library.GetWindowThreadProcessId.argtypes = [wintypes.HWND,
                                                 ctypes.POINTER(wintypes.DWORD)]
    library.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                     wintypes.WPARAM, wintypes.LPARAM]
    library.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
    library.GetWindow.restype = wintypes.HWND

    return library


def _dwm() -> ctypes.WinDLL:
    library = ctypes.WinDLL("dwmapi", use_last_error=True)

    library.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD,
                                              ctypes.c_void_p, wintypes.DWORD]

    return library


def _kernel32() -> ctypes.WinDLL:
    library = ctypes.WinDLL("kernel32", use_last_error=True)

    library.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    library.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    library.Process32FirstW.argtypes = [wintypes.HANDLE,
                                        ctypes.POINTER(_ProcessEntry)]
    library.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
    library.CloseHandle.argtypes = [wintypes.HANDLE]
    library.OpenProcess.restype = wintypes.HANDLE
    library.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    library.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    library.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                   wintypes.LPWSTR,
                                                   ctypes.POINTER(wintypes.DWORD)]
    library.GetWindowsDirectoryW.argtypes = [wintypes.LPWSTR, wintypes.UINT]

    return library


@dataclass(frozen=True)
class _Windows:
    """How this machine's own Windows is recognised: where it lives, and what its own
    files say about themselves.

    All three are read from the machine rather than written here. The names in
    particular: a file says which product it belongs to and who wrote it, and what those
    say is decided by the installation -- a machine set up in another language names its
    own product in that language.
    """

    home: str
    product: str
    company: str


def _windows_itself(kernel32: ctypes.WinDLL) -> _Windows:
    home = ctypes.create_unicode_buffer(_PATH)
    if not kernel32.GetWindowsDirectoryW(home, _PATH):
        raise UnreadableDevice("this machine would not say where Windows is installed")

    # The shell is Windows' own beyond doubt, so what it says about itself is what
    # everything else is compared against.
    shell = f"{home.value}\\explorer.exe"

    return _Windows(home=home.value.lower(),
                    product=_version_says(shell, "ProductName"),
                    company=_version_says(shell, "CompanyName"))


@dataclass(frozen=True)
class _Whose:
    """What the executable behind a process says about who it belongs to."""

    system: bool
    microsoft: bool
    publisher: str


def _whose(kernel32: ctypes.WinDLL, pid: Pid, windows: _Windows) -> _Whose:
    """Whether this process runs an executable of Windows itself, whether one of
    Microsoft's at all, and who it says wrote it. The second is the wider of the two
    and holds wherever the first does.

    A process that will not even open is Windows' own by that fact alone: the ones that
    refuse are the ones running as the system, and they are exactly the ones to leave
    alone.
    """
    handle = kernel32.OpenProcess(_QUERY_LIMITED, False, int(pid))
    if not handle:
        return _Whose(system=True, microsoft=True, publisher=windows.company)

    try:
        path = ctypes.create_unicode_buffer(_PATH)
        size = wintypes.DWORD(_PATH)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, path,
                                                   ctypes.byref(size)):
            return _Whose(system=True, microsoft=True, publisher=windows.company)

        publisher = _version_says(path.value, "CompanyName")
        if path.value.lower().startswith(windows.home):
            return _Whose(system=True, microsoft=True, publisher=publisher)

        return _Whose(
            system=(bool(windows.product)
                    and _version_says(path.value, "ProductName") == windows.product),
            microsoft=bool(windows.company) and publisher == windows.company,
            publisher=publisher)
    finally:
        kernel32.CloseHandle(handle)


def _version_says(path: str, about: str) -> str:
    """What this executable says about itself. Empty where it says nothing."""
    version = ctypes.WinDLL("version", use_last_error=True)

    size = version.GetFileVersionInfoSizeW(path, None)
    if not size:
        return ""

    block = ctypes.create_string_buffer(size)
    if not version.GetFileVersionInfoW(path, 0, size, block):
        return ""

    # Which language the strings are in is the file's own business, and it is written
    # in the file: the first translation is the one its strings are filed under.
    where = ctypes.c_void_p()
    length = wintypes.UINT()
    if not version.VerQueryValueW(block, "\\VarFileInfo\\Translation",
                                  ctypes.byref(where), ctypes.byref(length)):
        return ""
    language, page = ctypes.cast(where,
                                 ctypes.POINTER(wintypes.WORD * 2)).contents

    said = ctypes.c_wchar_p()
    if not version.VerQueryValueW(
            block, f"\\StringFileInfo\\{language:04x}{page:04x}\\{about}",
            ctypes.byref(said), ctypes.byref(length)):
        return ""

    return said.value or ""


def end(pid: Pid) -> None:
    """End this process outright, the way the task manager does.

    What `close` is not: nothing is asked and nothing can be said about unsaved work.
    It is for programs that have nothing on screen to ask -- measured, they do not
    answer WM_CLOSE and go on holding the card.

    A process that will not open, or is already gone, is left as it is: the screen is
    read again afterwards and says what actually happened.
    """
    _windows_only()

    kernel32 = _kernel32()
    handle = kernel32.OpenProcess(_TERMINATE, False, int(pid))
    if not handle:
        return

    try:
        kernel32.TerminateProcess(handle, 1)
    finally:
        kernel32.CloseHandle(handle)


def running() -> tuple[Running, ...]:
    """Every process: what it holds, what it has on screen, and whose it is."""
    _windows_only()

    held = _held()
    windowed = _windowed()
    kernel32 = _kernel32()
    windows = _windows_itself(kernel32)
    snapshot = _snapshot()
    whose = {pid: _whose(kernel32, pid, windows) for pid, _, _ in snapshot}

    return tuple(Running(pid=pid,
                         name=name,
                         parent=parent,
                         held=held.get(pid, Mib(0)),
                         windowed=pid in windowed,
                         system=whose[pid].system,
                         microsoft=whose[pid].microsoft,
                         publisher=whose[pid].publisher)
                 for pid, name, parent in snapshot)


def executing(name: str) -> frozenset[str]:
    """The directories the processes running this executable are running it from.

    Lowercased, because Windows does not tell two spellings of a path apart and the
    caller is comparing against directories read off the disk.

    A process this session may not open says nothing about where it runs from, and is
    left out. That is the same answer Windows itself gives for a process belonging to
    another account, and the directory it holds open refuses to be removed regardless.
    """
    _windows_only()

    kernel32 = _kernel32()
    wanted = name.lower()

    found = set()
    for pid, exe, _ in _snapshot():
        if exe.lower() != wanted:
            continue

        image = _image(kernel32, pid)
        if image:
            found.add(str(PurePath(image).parent).lower())

    return frozenset(found)


def processes(name: str) -> tuple[Pid, ...]:
    """Every process running this executable, whichever account started it.

    Listing one is not being able to end it: a router left behind by the scheduled task
    belongs to another logon session, and appears here so that a caller can say so
    rather than report success over a server that is still holding the card.
    """
    _windows_only()

    wanted = name.lower()
    return tuple(pid for pid, exe, _ in _snapshot() if exe.lower() == wanted)


@dataclass(frozen=True)
class Ran:
    """The elevated run finished: what it returned, and everything it wrote."""

    code: int
    said: str


@dataclass(frozen=True)
class Declined:
    """The prompt was answered no, or dismissed. Nothing ran."""


@dataclass(frozen=True)
class Unavailable:
    """Elevation could not be asked for at all, and the one line saying why.

    Windows refusing to raise the prompt is one way; an argument that cannot survive
    the trip through cmd is the other. Both leave the caller in the same place -- it
    holds the rights it had -- so both arrive as this rather than as an exception, and
    the caller's own message about running it elevated by hand covers them together.
    """

    why: str


Elevation = Ran | Declined | Unavailable


def elevate(command: Path, arguments: Sequence[str], working: Path) -> Elevation:
    r"""This same command, run again with a prompt for the rights it needs.

    Windows gives an elevated process a console of its own, which closes with it, so
    what it wrote would be gone before it could be read. It is therefore run through
    cmd with both streams redirected to a file, the window hidden, and the file read
    back here -- so that asking for rights costs a prompt and nothing else, and the
    answer arrives in the console the person is already looking at.

    Waited for rather than started and forgotten. The whole of the work happens over
    there, and a caller that returned before it finished would be reporting on nothing.
    """
    _windows_only()

    elevating = _Elevating(command=command, arguments=tuple(arguments),
                           working=working,
                           said=Path(tempfile.gettempdir())
                           / f"cm-elevated-{os.getpid()}.log")

    unpassable = _unpassable(elevating)
    if unpassable:
        return Unavailable(
            f"{unpassable[0]!r} cannot be handed to an elevated run: it holds a double "
            "quote, and cmd and the program behind it read one two different ways")

    shell32 = ctypes.WinDLL("shell32.dll", use_last_error=True)
    shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(_Elevated)]
    shell32.ShellExecuteExW.restype = wintypes.BOOL

    said = elevating.said
    asked = _Elevated()
    asked.size = ctypes.sizeof(_Elevated)
    asked.mask = _NO_CLOSE_PROCESS
    asked.verb = "runas"
    asked.file = "cmd.exe"
    asked.parameters = _through_cmd(elevating)
    asked.show = _HIDDEN

    try:
        if not shell32.ShellExecuteExW(ctypes.byref(asked)):
            failed = ctypes.get_last_error()
            if failed == _CANCELLED:
                return Declined()
            return Unavailable(f"Windows would not raise the prompt (error {failed})")

        return Ran(_awaited(asked.process), _wrote(said))
    finally:
        said.unlink(missing_ok=True)


@dataclass(frozen=True)
class _Elevating:
    """What to run with the rights, where to run it, and where its output goes."""

    command: Path
    arguments: tuple[str, ...]
    working: Path
    said: Path


_QUOTE = '"'


def _unpassable(elevating: _Elevating) -> tuple[str, ...]:
    """Whatever about this run cannot be handed to cmd, which is anything holding a
    double quote.

    Refused rather than escaped. Escaping one through cmd and then through the
    program's own parser is two conventions that disagree, and a Windows path cannot
    hold one anyway -- so what turns up here came off a command line, and saying so is
    better than passing on something that arrives over there as different text.
    """
    return tuple(one for one in _parts(elevating) if _QUOTE in one)


def _parts(elevating: _Elevating) -> tuple[str, ...]:
    """Everything about this run that reaches cmd as text."""
    return (str(elevating.command), *elevating.arguments,
            str(elevating.working), str(elevating.said))


def _through_cmd(elevating: _Elevating) -> str:
    r"""The command line cmd is handed, with both streams going to one file.

    The directory is changed here rather than passed as the shell's: the service that
    raises the prompt starts the elevated process in the system directory whatever it
    was told, and a command that runs `-m cm.autostart` from C:\Windows\System32 finds
    no package to run. Grouped in brackets so that a directory that has gone missing
    lands in the file with everything else instead of vanishing.

    cmd strips the outermost pair of quotes from what follows /c and leaves the rest
    alone, which is what lets every path inside keep quotes of its own.
    """
    written = " ".join(_shell_quoted(one)
                       for one in (str(elevating.command), *elevating.arguments))

    return (f'/c "(cd /d {_shell_quoted(str(elevating.working))} && {written}) '
            f'> {_shell_quoted(str(elevating.said))} 2>&1"')


def _shell_quoted(one: str) -> str:
    """One argument, as cmd has to be handed it.

    Whatever needs quotes gets them. Nothing here holds a double quote of its own:
    elevate refuses a run carrying one before any of this is built.
    """
    return one if not any(c in one for c in " \t&()[]{}^=;!'+,`~") else f'"{one}"'


def _awaited(process: wintypes.HANDLE) -> int:
    """What the elevated run returned, once it has returned it."""
    kernel32 = _kernel32()
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE,
                                            ctypes.POINTER(wintypes.DWORD)]

    try:
        kernel32.WaitForSingleObject(process, _FOREVER)
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(process, ctypes.byref(code)):
            return 1
        return int(code.value)
    finally:
        kernel32.CloseHandle(process)


def _wrote(said: Path) -> str:
    """What it wrote, or nothing where it wrote nothing at all."""
    try:
        return said.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def listening(port: Port) -> frozenset[Pid]:
    """Whose the sockets listening on this port are, over IPv4 and IPv6 both.

    Both, because which of the two a server ends up on is not something this decides:
    a router told to listen on every address takes IPv4, one told `::` takes IPv6, and
    a port held on either is a port a start cannot bind.

    A socket whose owner Windows will not name is reported as process 0 and left out.
    Nothing can be done to it, and it is not something a caller should be told to end.
    """
    _windows_only()

    iphlpapi = ctypes.WinDLL("iphlpapi.dll", use_last_error=True)

    found = {pid for family, row in ((_AF_INET, _TcpListener),
                                     (_AF_INET6, _Tcp6Listener))
             for pid in _owners(iphlpapi, family, row, port)}

    return frozenset(found - {Pid(0)})


def _owners(iphlpapi: ctypes.WinDLL, family: int, row: type[ctypes.Structure],
            port: Port) -> tuple[Pid, ...]:
    """The listeners of one address family, out of one table read whole.

    Read twice on purpose: the first call is what says how large the table is, and it
    can grow between the two, so a second insufficient buffer is taken as an answer of
    nothing rather than retried forever.
    """
    size = wintypes.DWORD(0)
    said = iphlpapi.GetExtendedTcpTable(None, ctypes.byref(size), False, family,
                                        _LISTENERS, 0)
    if said != _INSUFFICIENT_BUFFER or size.value == 0:
        return ()

    table = ctypes.create_string_buffer(size.value)
    said = iphlpapi.GetExtendedTcpTable(table, ctypes.byref(size), False, family,
                                        _LISTENERS, 0)
    if said != _NO_ERROR:
        return ()

    counted = wintypes.DWORD.from_buffer(table).value
    rows = (row * counted).from_buffer(table, ctypes.sizeof(wintypes.DWORD))

    return tuple(Pid(one.pid) for one in rows if _port_of(one.port) == port)


def _port_of(said: int) -> int:
    """A port as the table holds it: network byte order, in the low two bytes."""
    return ((said & 0xFF) << 8) | ((said >> 8) & 0xFF)


def _image(kernel32: ctypes.WinDLL, pid: Pid) -> str:
    """The executable behind this process, or "" where this session may not ask."""
    handle = kernel32.OpenProcess(_QUERY_LIMITED, False, int(pid))
    if not handle:
        return ""

    try:
        path = ctypes.create_unicode_buffer(_PATH)
        size = wintypes.DWORD(_PATH)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, path,
                                                   ctypes.byref(size)):
            return ""
        return path.value
    finally:
        kernel32.CloseHandle(handle)


def close(pid: Pid) -> None:
    """Ask this process's windows to close, the way clicking their corner would.

    Not a kill: an application that has something unsaved gets to say so. What that
    costs is that closing is a request, and the screen has to be read again to see
    whether it was granted.

    Only the ordinary windows are asked, which is why this is safe to do to Windows'
    own programs: a folder window closes and the shell behind it carries on.
    """
    _windows_only()

    user32 = _user32()
    for handle in _handles(user32, pid):
        user32.PostMessageW(handle, _WM_CLOSE, 0, 0)


def account() -> str:
    """Who this session is logged in as, in the form a task is registered under.

    DOMAIN\\user, which is what the scheduler resolves to an account. The environment
    carries something like it, and it is not asked: a variable can be set to anything,
    and a task registered under the wrong account is one that does not start at boot.
    """
    _windows_only()

    secur32 = ctypes.WinDLL("secur32.dll")

    size = wintypes.ULONG(0)
    secur32.GetUserNameExW(_SAM_COMPATIBLE, None, ctypes.byref(size))

    named = ctypes.create_unicode_buffer(size.value)
    if not secur32.GetUserNameExW(_SAM_COMPATIBLE, named, ctypes.byref(size)):
        raise UnreadableDevice("Windows did not say what this session is logged in as")

    return named.value


def elevated() -> bool:
    """Whether this session may change what the machine does for everyone on it."""
    _windows_only()

    return bool(ctypes.WinDLL("shell32.dll").IsUserAnAdmin())


def _windows_only() -> None:
    if sys.platform != "win32":
        raise UnreadableDevice(
            f"this reads what only Windows can say, and this is {sys.platform}")


def _held() -> dict[Pid, Mib]:
    """Dedicated video memory per process, out of the performance counters."""
    pdh = ctypes.WinDLL("pdh.dll")

    query = wintypes.HANDLE()
    if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)):
        raise UnreadableDevice("the performance counters would not open")

    try:
        counter = wintypes.HANDLE()
        # The English name, because the counter is called something else on a machine
        # installed in another language and the path is what is being asked for.
        if pdh.PdhAddEnglishCounterW(query, _COUNTER, 0, ctypes.byref(counter)):
            raise UnreadableDevice(f"no counter {_COUNTER} on this machine")
        if pdh.PdhCollectQueryData(query):
            raise UnreadableDevice("the performance counters returned nothing")

        return _counter_array(pdh, counter)
    finally:
        pdh.PdhCloseQuery(query)


def _counter_array(pdh: ctypes.WinDLL, counter: wintypes.HANDLE) -> dict[Pid, Mib]:
    size = wintypes.DWORD(0)
    count = wintypes.DWORD(0)

    status = pdh.PdhGetFormattedCounterArrayW(counter, _PDH_FMT_LARGE,
                                              ctypes.byref(size),
                                              ctypes.byref(count), None)
    if status & 0xFFFFFFFF != _PDH_MORE_DATA:
        # Asking with no room is how the size is learned; anything but "more data"
        # means the answer will not arrive on the second ask either.
        raise UnreadableDevice("the video memory counters could not be sized")

    buffer = ctypes.create_string_buffer(size.value)
    items = ctypes.cast(buffer, ctypes.POINTER(_CounterItem))
    if pdh.PdhGetFormattedCounterArrayW(counter, _PDH_FMT_LARGE, ctypes.byref(size),
                                        ctypes.byref(count), items):
        raise UnreadableDevice("the video memory counters could not be read")

    held: dict[Pid, Mib] = {}
    for index in range(count.value):
        item = items[index]
        found = _INSTANCE.match(item.name or "")
        if found is None or item.value.large <= 0:
            continue

        pid = Pid(int(found.group(1)))
        mib = Mib(round(item.value.large / _BYTES_PER_MIB))
        # One process appears once per adapter it draws on.
        held[pid] = Mib(held.get(pid, Mib(0)) + mib)

    return held


def _snapshot() -> tuple[tuple[Pid, str, Pid], ...]:
    """Every process: what it is called and whose child it is."""
    kernel32 = _kernel32()

    snapshot = kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if snapshot == wintypes.HANDLE(-1).value:
        raise UnreadableDevice("this machine would not list its processes")

    entry = _ProcessEntry()
    entry.size = ctypes.sizeof(_ProcessEntry)

    found = []
    try:
        more = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while more:
            found.append((Pid(entry.pid), entry.exe, Pid(entry.parent)))
            more = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)

    return tuple(found)


def _windowed() -> frozenset[Pid]:
    """Processes with an ordinary window: one a person would point at and call a window.

    Which is the same set alt-tab shows, and it is not a matter of taste. Windows draws
    a good deal that is visible and titled and is not a window in that sense: the input
    experience keeps one that the compositor never draws, the desktop itself is one, and
    every program with an icon in the notification area keeps its own out of sight.
    """
    user32 = _user32()
    dwm = _dwm()
    found: set[Pid] = set()

    def visit(handle, _):
        if _ordinary(user32, dwm, handle):
            found.add(_owner(user32, handle))
        return True

    user32.EnumWindows(_enum_windows()(visit), 0)
    return frozenset(found)


def _handles(user32: ctypes.WinDLL, pid: Pid) -> tuple[int, ...]:
    """The windows of this process that a person could close by their corner."""
    dwm = _dwm()
    found = []

    def visit(handle, _):
        if _ordinary(user32, dwm, handle) and _owner(user32, handle) == pid:
            found.append(handle)
        return True

    user32.EnumWindows(_enum_windows()(visit), 0)
    return tuple(found)


def _ordinary(user32: ctypes.WinDLL, dwm: ctypes.WinDLL, handle) -> bool:
    """Whether this is a window in the sense a person means by the word.

    Visible and titled, and then three things that sort the desktop's own furniture
    from what is on it. Not cloaked: a store app that has been closed leaves a window
    behind that nothing draws, and the input experience keeps one at all times -- asking
    that one to close is what left this machine without a keyboard. Not a tool window:
    that is the desktop itself and the taskbar. And owned by no other window, unless it
    says outright that it belongs in the taskbar, which is what tells a window from the
    dialogs and menus hanging off it.
    """
    if not user32.IsWindowVisible(handle):
        return False
    if user32.GetWindowTextLengthW(handle) == 0:
        return False
    if _cloaked(dwm, handle):
        return False

    style = user32.GetWindowLongW(handle, _GWL_EXSTYLE)
    if style & _WS_EX_TOOLWINDOW:
        return False

    return not user32.GetWindow(handle, _GW_OWNER) or bool(style & _WS_EX_APPWINDOW)


def _cloaked(dwm: ctypes.WinDLL, handle) -> bool:
    """Whether the compositor is drawing this window anywhere. It says so itself."""
    said = wintypes.DWORD()
    if dwm.DwmGetWindowAttribute(handle, _DWMWA_CLOAKED, ctypes.byref(said),
                                 ctypes.sizeof(said)):
        return False

    return bool(said.value)


def _owner(user32: ctypes.WinDLL, handle) -> Pid:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
    return Pid(pid.value)
