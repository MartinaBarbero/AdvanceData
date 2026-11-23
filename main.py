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

# Order matters: each notebook uses the files saved by the previous ones.
NOTEBOOKS = [
    "03_exploratory_analysis.ipynb",
    "04_model_table.ipynb",
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