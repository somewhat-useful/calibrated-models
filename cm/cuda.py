"""A CUDA version, which the release archives, the driver and the settings file all
speak in."""

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Cuda:
    """A CUDA version: what a build was compiled against, what a driver runs, what a
    settings file pins. The later version is the greater one, 13.10 above 13.4."""

    major: int
    minor: int

    @property
    def version(self) -> str:
        """As the archives spell it: 13.4."""
        return f"{self.major}.{self.minor}"
