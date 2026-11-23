"""Runs the whole project: executes the notebooks in order.

Usage, from the AdvanceData folder:
    python main.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "notebooks"

# The order follows the dependencies, not the numbers: notebooks 02 and 03 read
# the model table that notebook 04 builds, so 04 runs right after 01.
NOTEBOOKS = [
    "01_data_preparation.ipynb",
    "04_model_table.ipynb",
    "02_pace_gap_analysis.ipynb",
    "03_exploratory_analysis.ipynb",
    "05_walk_forward.ipynb",
    "06_race_prediction.ipynb",
]


def run_notebook(name):
    """Execute one notebook from top to bottom and save the outputs in it."""
    path = NOTEBOOK_DIR / name
    if not path.exists():
        sys.exit(f"Notebook not found: {path}")
    command = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        "--ExecutePreprocessor.timeout=-1",
        str(path),
    ]
    start = time.time()
    result = subprocess.run(command, cwd=NOTEBOOK_DIR)
    if result.returncode != 0:
        sys.exit(f"{name} failed, see the error above.")
    print(f"{name} done in {time.time() - start:.0f} seconds")


def main():
    total_start = time.time()
    for name in NOTEBOOKS:
        print(f"Running {name} ...")
        run_notebook(name)
    print(f"All notebooks finished in {(time.time() - total_start) / 60:.1f} minutes")


if __name__ == "__main__":
    main()