from pathlib import Path

import numpy as np

from aquaadapt.trajectory.tum import parse_tum


def test_tum_parser_sorts_normalizes_and_reports_malformed(tmp_path: Path) -> None:
    path = tmp_path / "track.tum"
    path.write_text(
        "# comment\n"
        "2 1 2 3 0 0 0 2\n"
        "bad row\n"
        "1 0 0 0 0 0 0 1\n"
        "3 0 0 0 0 0 0 0\n",
        encoding="utf-8",
    )
    trajectory = parse_tum(path)
    assert trajectory.timestamps.tolist() == [1.0, 2.0]
    assert np.allclose(np.linalg.norm(trajectory.quaternions, axis=1), 1)
    assert trajectory.malformed_lines == (3, 5)


def test_duplicate_timestamps_keep_first(tmp_path: Path) -> None:
    path = tmp_path / "track.tum"
    path.write_text("1 0 0 0 0 0 0 1\n1 9 9 9 0 0 0 1\n", encoding="utf-8")
    trajectory = parse_tum(path)
    assert len(trajectory.timestamps) == 1
    assert trajectory.duplicate_count == 1
    assert trajectory.translations[0].tolist() == [0, 0, 0]

