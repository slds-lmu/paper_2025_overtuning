# reshuffling

Note: you can obtain all raw data, including the original data from Nagler et al. (2024) and our HEBO with early stopping runs from:
https://doi.org/10.6084/m9.figshare.29248589.v1 (`reshuffling_raw_csvs.zip`)

## Steps:

1. Obtain `reshuffling_raw_csvs.zip`

```
echo "56ca45af9a1d4071d97e9780ebec19a8 reshuffling_raw_csvs.zip" | md5sum -c
```

2. Uncompress:

```
unzip reshuffling_raw_csvs.zip
```

3. Preprocess:

```
python preprocess.py
```

## Notes:

Each `csv` contains runs for different resamplings (holdout, repeated holdout, cross-validation and repeated cross-validation).
Columns should be self-explanatory.

## Citation:
```
@inproceedings{reshuffling,
  author = {Thomas Nagler and Lennart Schneider and Bernd Bischl and Matthias Feurer},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
  pages = {40486--40533},
  title = {Reshuffling Resampling Splits Can Improve Generalization of Hyperparameter Optimization},
  volume = {37},
  year = {2024}
}
```
