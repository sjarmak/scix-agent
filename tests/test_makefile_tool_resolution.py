import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fmt_check_uses_main_checkout_venv_from_linked_worktree(tmp_path: Path) -> None:
    main_checkout = tmp_path / "main"
    common_dir = main_checkout / ".git"
    bin_dir = main_checkout / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    for tool in ("black", "ruff"):
        (bin_dir / tool).touch()

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(f"#!/bin/sh\nprintf '%s\\n' '{common_dir}'\n")
    fake_git.chmod(0o755)

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    clean_environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "BLACK",
            "MAKEFLAGS",
            "MAKELEVEL",
            "MAKEOVERRIDES",
            "MFLAGS",
            "RUFF",
        }
    }
    result = subprocess.run(
        ["make", "-f", str(REPO_ROOT / "Makefile"), "-n", "fmt-check"],
        cwd=worktree,
        env={**clean_environment, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"{bin_dir / 'ruff'} check src/ scripts/ tests/" in result.stdout
    assert f"{bin_dir / 'black'} --check src/ scripts/ tests/" in result.stdout


def test_fmt_check_honors_tool_overrides(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "make",
            "-f",
            str(REPO_ROOT / "Makefile"),
            "-n",
            "fmt-check",
            "RUFF=/custom/ruff",
            "BLACK=/custom/black",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "/custom/ruff check src/ scripts/ tests/" in result.stdout
    assert "/custom/black --check src/ scripts/ tests/" in result.stdout
