# Formula 1 Cost Cap and Constructor Competitiveness

Individual project for the Advanced Data Analytics course, HEC Lausanne, autumn 2026.

## Research question

Did the 2021 cost cap reduce the competitive gap between F1 constructors?

## Data

Formula 1 World Championship (1950-2024), Kaggle, by Vopani (Rohan Rao).
Source: Ergast API. Downloaded CSVs are stored in `data/`.

## Repository structure
AdvanceData/
├── data/ raw CSV files from Kaggle
├── notebooks/ one notebook per pipeline stage
├── figures/ exported plots
└── README.md

## Notebooks

- `01_data_preparation.ipynb` — load data, build season-end standings (2018-2024), compute competitiveness gap metrics (std, range) normalized per race

## How to run

1. Open this workspace in Nuvolos, launch JupyterLab
2. Open a notebook from `notebooks/`
3. Run all cells top to bottom (each notebook must run from a fresh kernel)

**Note on paths:** the Jupyter kernel in this workspace always starts from `/files` (the workspace root), not from the notebook's own folder. All file paths in the notebooks are therefore written relative to `/files`, e.g. `AdvanceData/data/races.csv`, not `../data/races.csv`.

## AI tool disclosure

Used Claude (Anthropic) for: debugging file path and kernel errors, code review, and drafting this README. All analysis logic and interpretation written/reviewed by the author.