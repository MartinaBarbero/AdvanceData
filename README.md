# Predicting Formula 1 race winners before 2021 and after the cost cap

This project asks whether the winner of a Formula 1 race can be predicted using only information known before the start, and whether a model trained before the 2021 cost cap still works after it. The unit of observation is one driver in one race (2014 to 2026), and the main model is a logistic regression evaluated chronologically, one season at a time.

## Install

Python 3.12 (the version in the Nuvolos workspace). From the `AdvanceData` folder:

```
pip install -r requirements.txt
```

## Run

One command runs everything, in the right order:

```
python main.py
```

It executes the notebooks in `notebooks/` one after the other. You can also open and run them by hand, each one from top to bottom in a fresh kernel. The order matters because later notebooks read what earlier ones save:

| Order | Notebook | What it does |
|---|---|---|
| 1 | `01_data_preparation.ipynb` | Checks the raw files, final standings, competitiveness over the seasons, cost cap levels |
| 2 | `04_model_table.ipynb` | Builds the driver-by-race table with the pre-race features |
| 3 | `02_pace_gap_analysis.ipynb` | Team pace gap to the fastest team, strongest teams against the rest |
| 4 | `03_exploratory_analysis.ipynb` | Winners and wins by driver and season |
| 5 | `05_walk_forward.ipynb` | Fits the logistic regression season by season and evaluates it |
| 6 | `06_race_prediction.ipynb` | Win probabilities for all drivers of a chosen race |

Code used in more than one notebook is in `src/`.

## Time and output

The whole pipeline takes about 1 minute in the Nuvolos workspace (notebook 05 is the slowest, around 30 seconds). No resource beyond the workspace is needed.

Figures are written to `figures/` and tables to `tables/`. The executed notebooks keep their outputs, so they can be read without rerunning.

## Data

- **Formula 1 results, races, lap times, constructors and standings, 2014 to 2024:** Kaggle Formula 1 dataset, stored in `data/`. TODO: add the dataset name, author and licence from its Kaggle page.
- **Races from 2025 and 2026:** Jolpica-F1 API (https://api.jolpi.ca/ergast/f1/). TODO: add the date the data was downloaded and the usage terms from the Jolpica page.
- **Cost cap levels:** entered by hand in `data/external/cost_cap.csv`, with the source of each value (FIA Financial Regulations and Formula1.com) in the file itself. The 2026 value (215 million USD) uses a different scope of costs, so it is not comparable with 2021 to 2025.

Cleaning decisions (missing values, renamed teams, minimum laps for the pace measure) are documented in the notebooks where they are made.

## Reproducibility

No absolute paths are used. The model has no random component, so a rerun gives the same numbers.

## AI Disclolsure
Used Claude (Anthropic) for: debugging file path and kernel errors, code review, and drafting this README. All analysis logic and interpretation written/reviewed by the author.