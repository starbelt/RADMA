from pathlib import Path
import os

def get_repo_root() -> Path:
    """Return Coral-TPU-Characterization repo root, independent of machine layout."""
    # Prefer environment variable if user set it
    if "CORAL_REPO" in os.environ:
        return Path(os.environ["CORAL_REPO"]).expanduser().resolve()

    # Default search locations
    candidates = [
        Path.home() / "Coral-TPU-Characterization",
        Path.home() / "Dev/repos/Coral-TPU-Characterization",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError("Coral-TPU-Characterization repo not found. "
                            "Set CORAL_REPO environment variable to point to it.")
