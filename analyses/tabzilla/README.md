# TabZilla

## Steps:

1. Obtain `metadataset_clean.csv` from https://drive.google.com/drive/folders/1cHisTmruPHDCYVOYnaqvTdybLngMkB8R

```
echo "f287d8d81333da2660a612137ff662d4 metadataset_clean.csv" | md5sum -c
```

2. Preprocess:

```
python preprocess.py
```

Output is a `csv` labelled `data.csv` with columns that should be self-explanatory.

## Notes:


## Citation:

```
@inproceedings{tabzilla,
  author = {Duncan McElfresh and Sujay Khandagale and Jonathan Valverde and Vishak {Prasad C.} and Ganesh Ramakrishnan and Micah Goldblum and Colin White},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
  pages = {76336--76369},
  title = {When Do Neural Nets Outperform Boosted Trees on Tabular Data?},
  volume = {36},
  year = {2023}
}
```
