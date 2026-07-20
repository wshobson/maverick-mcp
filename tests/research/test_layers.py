"""The layer contracts are live and provably enforceable."""

import shutil
import subprocess


def test_import_contracts_pass():
    exe = shutil.which("lint-imports")
    assert exe, "lint-imports not on PATH (dev group not installed?)"
    result = subprocess.run([exe], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 broken" in result.stdout
