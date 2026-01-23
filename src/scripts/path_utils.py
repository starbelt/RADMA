from pathlib import Path
import os
import sys

def get_repo_root() -> Path:
    """
    Returns the repository root by climbing up from the current file's location.
    Stops when it finds a marker (like '.git', 'libs', or 'results').
    """
    # Prefer environment variable if set
    if "CORAL_REPO" in os.environ:
        return Path(os.environ["CORAL_REPO"]).expanduser().resolve()

    # Start from the location of THIS file (path_utils.py)
    current_path = Path(__file__).resolve()

    # Climb up the directory tree looking for root markers
    for parent in [current_path] + list(current_path.parents):
        if (parent / ".git").exists() or \
        ((parent / "libs").exists() and (parent / "results").exists()):
            return parent

    # Fallbacks
    cwd = Path.cwd()
    if (cwd / "libs").exists() and (cwd / "results").exists():
        return cwd

    candidates = [
        Path.home() / "Coral-TPU-Characterization",
        Path.home() / "CoralGUI", 
        Path.home() / "Dev/repos/Coral-TPU-Characterization",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "Could not auto-detect repository root. "
    )