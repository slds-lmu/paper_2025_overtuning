# WDTB

## Steps:

1. Obtain `tabular_benchmark.csv`

```
wget -O tabular_benchmark.csv https://figshare.com/ndownloader/files/40081681
echo "73159b346ec355e4cebbc7702ac4f91a tabular_benchmark.csv" | md5sum -c
```

2. Preprocess:

```
python preprocess.py
```

Output is a `csv` labelled `data.csv` with columns that should be self-explanatory.

## Notes:


## Citation:

```
@article{tabularbenchmark,
  author = {Léo Grinsztajn and Edouard Oyallon and Gaël Varoquaux},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
  pages = {507--520},
  title = {Why do tree-based models still outperform deep learning on typical tabular data?},
  volume = {35},
  year = {2022}
}
```
