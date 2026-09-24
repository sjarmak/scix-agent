from pathlib import Path

CRON_EXAMPLES = Path(__file__).parents[1] / "scripts" / "cron"
SCIX_BATCH_MARKERS = ("scix-batch", "SCIX_BATCH")
XDG_RUNTIME_DIR = "XDG_RUNTIME_DIR=/run/user/1000"


def test_scix_batch_cron_lines_set_xdg_runtime_dir() -> None:
    invalid_lines: list[str] = []

    for cron_file in sorted(CRON_EXAMPLES.glob("*.cron.example")):
        for line_number, line in enumerate(cron_file.read_text().splitlines(), start=1):
            stripped = line.strip()
            if (
                stripped
                and not stripped.startswith("#")
                and any(marker in stripped for marker in SCIX_BATCH_MARKERS)
            ):
                if XDG_RUNTIME_DIR not in stripped:
                    invalid_lines.append(f"{cron_file.name}:{line_number}: {stripped}")

    failure_message = f"cron lines invoking scix-batch must set {XDG_RUNTIME_DIR}:\n" + "\n".join(
        invalid_lines
    )
    assert not invalid_lines, failure_message
