"""Research Tree: a Git-native inquiry graph."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("research-tree")
except PackageNotFoundError:  # source checkout
    __version__ = "0.4.0"
