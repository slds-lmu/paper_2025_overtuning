# TabRepo

## Steps:

1. Obtain `tabrepo`:

```
git clone https://github.com/autogluon/tabrepo.git tabrepogh
```

2. Install it within your virtualenv:

```
cd tabrepogh
pip install -e .
```

3. Preprocess:

```
python preprocess.py
```

Output is a `csv` labelled `data.csv` with columns that should be self-explanatory.


## Notes:

Currently uses "D244_F3_C1530_175" TabRepo context.

## Citation:

```
@inproceedings{tabrepo,
  title = {{TabRepo}: A Large Scale Repository of Tabular Model Evaluations and its {AutoML} Applications},
  author = {David Salinas and Nick Erickson},
  booktitle = {Proceedings of the Third International Conference on Automated Machine Learning},
  pages = {19/1--30},
  year = {2024},
  editor = {K. Eggensperger and R. Garnett and J. Vanschoren and M. Lindauer and J. R. Gardner},
  volume = {256}
}
```
