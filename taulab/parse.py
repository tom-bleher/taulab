"""Folder parsers — walk a directory of CSVs/xlsx and return one
:class:`ParseResult` per file.

Upstream taulab exposes ``parse_csv_folder`` / ``parse_excel_folder`` under
``taulab.parse``; this module preserves that import path.  The key convenience
is **filename-as-metadata**: named regex groups
(``(?P<wavelength>\\d+)nm``) are captured into each result's
``metadata`` dict, so a folder of ``sweep_532nm_100mA.csv`` files becomes a
ready-to-concat list with ``wavelength`` / ``current`` columns reconstructable
from provenance.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .datatypes import ParseResult
from .io import read_table

__all__ = ["parse_csv_folder", "parse_excel_folder", "parse_filename_metadata"]


def parse_filename_metadata(filename: str, pattern: str) -> dict[str, Any]:
    """Return ``re.search(pattern, filename).groupdict()`` or ``{}`` on no match.

    Values are strings; callers coerce to int / float as needed.  Use named
    capture groups to get a self-describing dict::

        >>> parse_filename_metadata('sweep_532nm_100mA.csv',
        ...                         r'sweep_(?P<wavelength>\\d+)nm_(?P<current>\\d+)mA')
        {'wavelength': '532', 'current': '100'}
    """
    m = re.search(pattern, filename)
    return m.groupdict() if m else {}


def _parse_folder(folder, glob_patterns: tuple[str, ...],
                  filename_pattern: str | None,
                  read_kwargs: dict) -> list[ParseResult]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"parse folder not found: {folder}")
    paths: list[Path] = []
    for pat in glob_patterns:
        paths.extend(folder.glob(pat))
    paths.sort()
    out: list[ParseResult] = []
    for path in paths:
        if filename_pattern is not None:
            meta = parse_filename_metadata(path.name, filename_pattern)
            if not meta:
                continue                               # filename didn't match -> skip
        else:
            meta = {}
        df = read_table(path, **read_kwargs)
        out.append(ParseResult(data=df, metadata=meta, source=str(path)))
    return out


def parse_csv_folder(folder, *, filename_pattern: str | None = None,
                     **read_kwargs) -> list[ParseResult]:
    """Walk ``folder`` for ``*.csv`` files, parse each into a :class:`ParseResult`.

    ``filename_pattern`` is an optional regex applied to each file's *basename*
    with :func:`re.search`; non-matching files are skipped and matching files
    contribute their ``groupdict()`` to ``ParseResult.metadata``.

    Extra keyword arguments forward to :func:`taulab.io.read_table`.
    """
    return _parse_folder(folder, ("*.csv",), filename_pattern, read_kwargs)


def parse_excel_folder(folder, *, filename_pattern: str | None = None,
                       **read_kwargs) -> list[ParseResult]:
    """Walk ``folder`` for ``*.xlsx`` / ``*.xls`` files, parse each."""
    return _parse_folder(folder, ("*.xlsx", "*.xls"), filename_pattern, read_kwargs)
