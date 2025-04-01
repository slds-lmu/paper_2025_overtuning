This folder contains all code needed to perform the reanalyses of published HPO study.

Note:  You will need at least 32GB of RAM.

Each folder of each study contains a README with instructions how to obtain data and how to preprocess data if needed.

It is best to create a separate python 3.10.6 virtual environment, then install requirements:

```
pip install -r requirements.txt
```

Then

```
python create_overtuning_postprocessings.py
```

will create a bash script for each HPO study, e.g., `run_overtuning_postprocessing_fcnet.sh`.
These scripts repeatedly make use of `overtuning_postprocessing.py` for the actual postprocessing.

Outputs are written as `csv` files to `csvs/`.

To make replication easier, we provide `overtuning_csvs.zip` via:
https://doi.org/10.6084/m9.figshare.29248589.v1

Download and unzip so that files are placed into `csvs/`.

Afterwards, the actual analyses can be performed, see:
* `figure_1.ipynb`
* `figures_section_5.ipynb`
* `figures_section_6.ipynb`
* `analyses_section_6.ipynb` (you need an R kernel, see below, and fitting the models can take quite some time)

To obtain the WDTB data needed for Figure 1, follow the descriptions in the README of `wdtb/`.

If you have troubles to set your active virtual environment as a kernel for the Jupyter notebooks:
```
python -m pip install jupyter
python -m ipykernel install --user --name=overtuning
```

For the mixed models we used R 4.4.2 with packages:
* data.table 1.17.0
* mlr3misc 0.16.0
* lme4 1.1-35.5
* MuMIn 1.48.4
* lmerTest 3.1-3
* dplyr 1.1.4
* stringr 1.5.1

To obtain an R kernel, install R, then:

```
install.packages("IRkernel")
IRkernel::installspec()
```

