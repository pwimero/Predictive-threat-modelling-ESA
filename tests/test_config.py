"""Regression tests for portable local data-path resolution."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mirage.config import _resolve_local_path


class LocalPathResolutionTests(unittest.TestCase):
    def test_prefers_declared_path_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "data" / "mission1"
            local = root / "data.nosync" / "mission1"
            canonical.mkdir(parents=True)
            local.mkdir(parents=True)

            self.assertEqual(_resolve_local_path(canonical), canonical)

    def test_uses_nosync_directory_when_canonical_path_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "data" / "mission1" / "raw"
            local = root / "data.nosync" / "mission1" / "raw"
            local.mkdir(parents=True)

            self.assertEqual(_resolve_local_path(canonical), local)

    def test_uses_nosync_archive_name_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "ESA-Mission1.zip"
            local = root / "ESA-Mission1.zip.nosync.zip"
            local.touch()

            self.assertEqual(_resolve_local_path(canonical, archive=True), local)


if __name__ == "__main__":
    unittest.main()
