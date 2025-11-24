# Tunisian Arabic LLM Evaluation (TUNIZI)

[![arXiv](https://img.shields.io/badge/arXiv-2511.16683-b31b1b.svg)](https://arxiv.org/abs/2511.16683)

This repository contains scripts, datasets, and helper code used to evaluate several large language models (LLMs) on tasks involving Tunisian Arabic: transliteration/normalization evaluation, translation similarity, and sentiment classification. The project collects model outputs (stored in the `task 1/`, `task 2/`, and `task 3/` folders), computes standard metrics, and exports results as CSV and LaTeX tables.

## Repository layout

- `dataset.csv` - main dataset (semicolon-separated) used as ground truth for the tasks.
- `dataset.py` - orchestrates evaluation across Task 1 (transliteration), Task 2 (translation), and Task 3 (sentiment classification). Computes metrics and writes per-task CSV summaries.
- `distribution.py` - small script to visualize sentiment class distribution using Matplotlib/Seaborn.
- `tunizi_to_tn_ar_similarity.py` - utility functions used for string similarity metrics on Arabic/Tunisian text (CER, Levenshtein, LCS).
- `csv_to_latex.py` - converts the Task 3 classification metrics CSV into a LaTeX table and writes `sentiment_table.tex`.
- `sentiment_table.tex` - generated LaTeX table (committed here as an example / output).
- `review.txt`, `tunizi.txt`, `dataset.txt` - auxiliary text files used for reference.
- `task 1/`, `task 2/`, `task 3/` - folders containing prompts, raw model outputs (one file per LLM), and generated metrics CSVs. Example output CSVs:
  - `task 1/task1_metrics.csv`
  - `task 2/task2_translation_metrics.csv`
  - `task 3/task3_classification_metrics.csv`


## Python environment & dependencies

The scripts are written for Python 3.8+ and rely on common scientific and NLP packages. Key dependencies observed in the code:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk
- python-Levenshtein
- bert-score (package name `bert-score`)

Install dependencies (recommended in a virtual environment). Example (Windows / cmd.exe):

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn nltk python-Levenshtein bert-score
```

Notes:
- If you prefer a `requirements.txt`, create one and run `pip install -r requirements.txt`.
- The code uses NLTK tokenizers and other resources. Run these once to download required NLTK data (in `cmd.exe`):

```cmd
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Quick usage

All commands below assume you're in the repository root and that your Python environment is active.

- Generate task metrics (runs the orchestrator `dataset.py` which processes `task 1/`, `task 2/`, `task 3/` folders and writes CSV results):

```cmd
python dataset.py
```

- Create a visualization of sentiment distribution (opens or saves a plot depending on your environment):

```cmd
python distribution.py
```

- Convert `task 3` metrics CSV into a LaTeX table (writes `sentiment_table.tex`):

```cmd
python csv_to_latex.py
```

Notes on tasks folders:
- Each `task X/` folder contains raw model outputs (one per LLM). The orchestrator reads these files, compares to the ground truth in `dataset.csv`, and writes per-task metrics CSVs into the same folder.

## Expected inputs and outputs

- Inputs: `dataset.csv` (ground truth); per-LLM output text files inside `task 1/`, `task 2/`, `task 3/` (one prediction per line).
- Outputs: per-task metrics CSVs (e.g. `task3_classification_metrics.csv`), plots (from `distribution.py`), and `sentiment_table.tex` (LaTeX table for Task 3 results).

## Implementation notes & assumptions

- The evaluation code expects the dataset CSV to have specific column names such as `tunisian_arabic`, `english_translation`, and `sentiment`. Confirm column names in `dataset.csv` before running.
- Some scripts refer to absolute paths for task folders; you may need to update `Path(...)` values in `dataset.py` to match your local layout if you move the repository.
- The NLTK downloads in the repository appear to include a non-standard key (`punkt_tab`) in one place; use `'punkt'` as shown above if you encounter errors.

## Troubleshooting

- If you hit import errors, ensure the environment is activated and packages are installed.
- If NLTK tokenizers raise missing resource errors, run the `nltk.download(...)` command shown above.
- If `python-Levenshtein` fails to install on Windows, ensure you have a build toolchain or use the binary wheels available on PyPI for your Python version.

## Contributing

If you'd like to contribute to this project, open an issue or submit a pull request or contact me through my email *mohamed.mahdi@etudiant-enit.utm.tn*.

## License

This repository does not include a license file. Add a `LICENSE` file if you want to define reuse terms.

## Contact

If you need help running the scripts or want changes to the README, add an issue or message the repository owner.

---
Generated on November 12, 2025 — README crafted from repository sources.
