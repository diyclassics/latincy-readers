"""Corpus reader implementations."""

from latincyreaders.readers.tesserae import TesseraeReader
from latincyreaders.readers.plaintext import PlaintextReader, LatinLibraryReader
from latincyreaders.readers.tei import TEIReader, PerseusReader
from latincyreaders.readers.ud import UDReader

__all__ = [
    "TesseraeReader",
    "PlaintextReader",
    "LatinLibraryReader",
    "TEIReader",
    "PerseusReader",
    "UDReader",
]
