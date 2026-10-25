import sys

from ml_lab.process import run_at_full_speed


def test_full_speed_is_requested_on_windows_and_ignored_elsewhere() -> None:
    assert run_at_full_speed() is (sys.platform == "win32")
