This folder is based upon the code base of Nagler et al. (2024) and can be used to perform the HEBO with early stopping runs analyzed in the paper.

Original code base: https://github.com/slds-lmu/paper_2024_reshuffling released under MIT License


## Installation

Create a python 3.10.6 virtual environment, then install the package and the following dependencies:

```
pip install -e .
pip install "gpytorch>=1.4.0"
pip install "pymoo>=0.6.0"
pip install "HEBO==0.3.5" --no-deps
pip install "smac==2.2.0"
```

While the code base here can in principle be used to replicate all HPO runs of Nagler et al. (2024), we only used it to perform the HEBO with early stopping runs (5-fold CV, ROC AUC).

## Experiments:

* create experiment scripts and run them (`run_experiments.sh`), e.g., via slurm submit scripts
* this will create folders and result files in `results/`

Below is example code to generate experiments (e.g., for CatBoost).
See `main.py` and the main logic in `overtunebench`.
Code in `analyze/` is used to analyze experiment results.

### HEBO with early stopping 5-fold CV

`python create_experiments.py --classifier=catboost --default=False --optimizer=hebo_makarova --valid_type=cv --n_repeats=1`

## Analysis:

* create analysis scripts and run them (`run_analysis.sh`) within the `analyze` directory, e.g. via slurm submit scripts
* this will create folders in `csvs/raw/`

### HEBO with early stopping 5-fold CV

`python create_analyses.py --optimizer=hebo_makarova --valid_type=cv --n_repeats=1 --type=basic --max_workers=1 --reshuffle=Both --check_files=False`

## Collect Results:

* collect analyzed results
* this will create result files in `csvs/`

`python collect_results.py --valid_type=cv`
