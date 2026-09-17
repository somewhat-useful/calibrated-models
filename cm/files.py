"""Reading and writing files. No decision is taken here."""

import shutil
import zipfile
from datetime import datetime
from pathlib import Path, PurePath

from .units import Bytes


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_utf16(path: Path, text: str) -> None:
    """A file for a Windows program that reads UTF-16, byte order mark and all."""
    path.write_text(text, encoding="utf-16")


def exists(path: Path) -> bool:
    return path.exists()


def replace(path: Path, text: str) -> None:
    """Write a file that belongs to another program, whole or not at all.

    Through a temporary and moved into place: a half-written configuration leaves that
    program unable to start, and this runs while it may be open.
    """
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def backup(path: Path, keep: int) -> Path:
    """A dated copy beside the file, keeping no more than `keep` of them.

    Dated rather than one rolling copy: the version worth recovering is rarely the one
    immediately before the last run, and a single .bak is overwritten by the second run
    with the good version gone. Milliseconds, because two runs inside one second is not
    hypothetical.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    copy = path.with_name(f"{path.name}.bak-{stamp}")
    shutil.copyfile(path, copy)

    for old in sorted(path.parent.glob(f"{path.name}.bak-*"), reverse=True)[keep:]:
        old.unlink()

    return copy


def directories(path: Path) -> tuple[str, ...]:
    """The names of the directories directly under this one."""
    if not path.is_dir():
        return ()
    return tuple(entry.name for entry in path.iterdir() if entry.is_dir())


def contents(path: Path) -> tuple[Path, ...]:
    """Everything directly under this directory."""
    if not path.is_dir():
        return ()
    return tuple(path.iterdir())


def named(path: Path, suffix: str) -> tuple[str, ...]:
    """The names of the files directly under this directory ending in this suffix."""
    return tuple(entry.name for entry in contents(path)
                 if entry.is_file() and entry.suffix.lower() == suffix)


def under(root: Path, suffix: str) -> tuple[PurePath, ...]:
    """Where every file ending in this suffix sits under this directory, at any depth
    and relative to it. Sorted, so a library walked twice comes back the same."""
    if not root.is_dir():
        return ()

    return tuple(sorted((PurePath(found.relative_to(root)) for found in root.rglob("*")
                         if found.is_file() and found.suffix.lower() == suffix),
                        key=str))


def length(path: Path) -> Bytes:
    """How big one file is, in bytes."""
    return Bytes(path.stat().st_size)


def head(path: Path, count: Bytes) -> bytes:
    """The first bytes of a file, or all of it where it is shorter than that.

    For files nothing here reads whole: a model is gibibytes of weights behind a header
    of kilobytes, and the header is all this program has ever needed of one.
    """
    with path.open("rb") as handle:
        return handle.read(count)


def size(path: Path) -> Bytes:
    """How much there is under this directory, in bytes."""
    return Bytes(sum(entry.stat().st_size
                     for entry in path.rglob("*") if entry.is_file()))


def remove(path: Path) -> None:
    """A directory and everything under it. Absent is the same as removed."""
    if path.is_dir():
        shutil.rmtree(path)


def delete(path: Path) -> None:
    """A file that has stopped being true. Absent is the same as deleted."""
    path.unlink(missing_ok=True)


def discard(path: Path) -> None:
    """Scratch space that is no longer wanted, whether or not it will go.

    A temporary left behind because something else has it open is not a reason to fail
    a run that has otherwise finished.
    """
    shutil.rmtree(path, ignore_errors=True)


def ensure(path: Path) -> None:
    """This directory, whether or not it was there. Whatever is in it stays."""
    path.mkdir(parents=True, exist_ok=True)


def tail(path: Path, lines: int) -> str:
    """The end of a file, for a log worth reading after something failed.

    Undecodable bytes are replaced rather than raised: a log is shown because a start
    failed, and failing to show it over one byte would leave nothing to go on.
    """
    if not path.is_file():
        return ""

    read = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(read[-lines:])


def fresh(path: Path) -> None:
    """An empty directory here, whatever was here before."""
    remove(path)
    path.mkdir(parents=True)


def move(source: Path, destination: Path) -> None:
    shutil.move(str(source), str(destination))


def copy(source: Path, into: Path) -> None:
    shutil.copy2(source, into / source.name)


def unpack(archive: Path, into: Path) -> None:
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(into)
