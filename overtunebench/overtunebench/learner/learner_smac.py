import json
import os
import random
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import openml
import optuna
import pandas as pd
from optuna.trial import Trial
from sklearn.datasets import make_classification, make_hastie_10_2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from overtunebench.algorithms.classifier import Classifier
from overtunebench.samplers import SmacSampler
from overtunebench.utils import unify_missing_values


class LearnerSmac(ABC):
    def __init__(
        self,
        classifier: Classifier,
        metric: str,
        data_id: int,
        valid_type: str,
        train_valid_size: int,
        reshuffle: bool,
        valid_frac: Union[float, None],
        n_splits: Union[int, None],
        n_repeats: Union[int, None],
        test_size: int,
        add_valid_size: int,
        n_trials: int,
        seed: int,
        results_path: str,
        cv_metric_to_metric: Union[dict, None],
        write_results_to_disk: bool = False,
        bootstrap_test: bool = True,
    ):
        self.metrics = ["auc"]
        self.metrics_direction = {
            "auc": "maximize",
        }
        assert metric in self.metrics
        self.metric = metric
        self.cv_metric_to_metric = cv_metric_to_metric

        self.classifier = classifier
        self.data_id = data_id
        self.valid_type = valid_type
        self.train_valid_size = train_valid_size
        self.reshuffle = reshuffle
        self.valid_frac = valid_frac
        self.valid_size = (
            int(self.train_valid_size * self.valid_frac)
            if self.valid_frac is not None
            else None
        )
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.test_size = test_size
        self.add_valid_size = add_valid_size
        self.n_trials = n_trials
        self.seed = seed
        self.results_path = results_path
        self.file_name = f"study_seed_{self.seed}"
        self.write_results_to_disk = write_results_to_disk
        self.bootstrap_test = bootstrap_test

        self.label_encoder = None

        self.has_missings = False
        self.labels = None
        self.n_classes = None
        self.multiclass = False
        self.cat_features = None
        self.num_features = None
        self.x_valid_train = None
        self.x_test = None
        self.x_train = None
        self.x_valid = None
        self.y_valid_train = None
        self.y_test = None
        self.y_train = None
        self.y_valid = None
        self.y_train_hist = None
        self.y_valid_hist = None
        self.y_pred_train_proba_hist = None
        self.y_pred_valid_proba_hist = None
        self.y_pred_test_proba_hist = None
        self.y_pred_valid_train_proba_hist = None
        self.y_pred_test_proba_retrained_hist = None

        self.cv = None
        self.cv_splits = None
        self.cv_splits_hist_train = None
        self.cv_splits_hist_valid = None
        self.train_size = None

        self.x_add_valid_use = None
        self.y_add_valid_use = None
        self.x_add_valid = None
        self.y_add_valid = None
        self.y_add_valid_hist = None
        self.y_pred_add_valid_proba_hist = None

        self.cv_add_valid = None
        self.cv_splits_add_valid = None
        self.cv_splits_add_valid_hist_valid = None

        self.study = None
        self.study_name = None
        self.storage = None

    def prepare_data(self) -> None:
        """
        Prepare the data for the optimization.
        """

        if self.data_id == 99999:
            # generate data for testing
            x, y = make_hastie_10_2(
                n_samples=self.train_valid_size
                + self.test_size
                + self.add_valid_size
                + 1000,
                random_state=0,
            )
            x = pd.DataFrame(x)
            x.columns = [f"feature_{i}" for i in range(x.shape[1])]
            y = y.astype(int)
            y = pd.Series(y, dtype="category")
            y.columns = ["target"]
            categorical_indicator = [False] * x.shape[1]
        elif self.data_id == 11111:
            # generate data for testing
            x, y = make_classification(
                n_samples=self.train_valid_size
                + self.test_size
                + self.add_valid_size
                + 1000,
                n_features=20,
                n_informative=10,
                n_redundant=2,
                n_repeated=2,
                n_classes=2,
                n_clusters_per_class=4,
                weights=None,
                flip_y=0.1,
                class_sep=1.0,
                hypercube=True,
                shift=0.0,
                scale=1.0,
                shuffle=True,
                random_state=0,
            )
            x = pd.DataFrame(x)
            x.columns = [f"feature_{i}" for i in range(x.shape[1])]
            y = y.astype(int)
            y = pd.Series(y, dtype="category")
            y.columns = ["target"]
            categorical_indicator = [False] * x.shape[1]
        else:
            # download the OpenML data for the dataset
            dataset = openml.datasets.get_dataset(
                self.data_id,
                download_data=True,
                download_qualities=False,
                download_features_meta_data=False,
            )

            # get the pandas dataframe
            x, y, categorical_indicator, _ = dataset.get_data(
                dataset_format="dataframe",
                target=dataset.default_target_attribute,
            )

        # check if dataset is large enough for train_valid_size, test_size and add_valid_size
        if (
            x.shape[0]
            < self.train_valid_size + self.test_size + self.add_valid_size + 1000
        ):
            raise ValueError(
                "Dataset too small for train_valid_size, test_size and add_valid_size"
            )

        # check if feature names contain "__"
        if any("__" in col for col in x.columns):
            raise ValueError(
                'Feature names contain "__" which is not allowed due to preprocessing'
            )

        # convert categorical columns to object dtype
        for col in range(len(categorical_indicator)):
            if categorical_indicator[col]:
                x[x.columns[col]] = x[x.columns[col]].astype("object")
        self.cat_features = [
            x.columns.get_loc(col) for col in x.columns if x[col].dtype.name == "object"
        ]
        self.num_features = list(set(range(x.shape[1])) - set(self.cat_features))

        # check if missing values are present
        # if yes bring missing values into a standardized format
        if x.isnull().values.any():
            self.has_missings = True
            x = unify_missing_values(x)

        if y.isnull().values.any():
            raise ValueError("Missing values in target column")

        # map the labels to integers
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        self.labels = np.unique(y).tolist()
        self.n_classes = len(self.label_encoder.classes_)
        if self.n_classes > 2:
            self.multiclass = True

        # split the data into used and unused data
        x_not_use, x_use, y_not_use, y_use = train_test_split(
            x,
            y,
            test_size=self.train_valid_size + self.test_size,
            random_state=self.seed,
            stratify=y,
        )
        # split the used data into valid_train and test
        x_valid_train, x_test, y_valid_train, y_test = train_test_split(
            x_use,
            y_use,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=y_use,
        )
        # from the unused data sample add_valid_size data to use as additional validation data
        _, x_add_valid_use, _, y_add_valid_use = train_test_split(
            x_not_use,
            y_not_use,
            test_size=self.add_valid_size,
            random_state=self.seed,
            stratify=y_not_use,
        )

        self.x_valid_train = x_valid_train
        self.x_add_valid_use = x_add_valid_use
        self.x_test = x_test
        self.y_train_hist = []
        self.y_valid_hist = []
        self.y_add_valid_hist = []
        self.y_pred_train_proba_hist = []
        self.y_pred_valid_proba_hist = []
        self.y_pred_test_proba_hist = []
        self.y_pred_add_valid_proba_hist = []
        self.y_pred_valid_train_proba_hist = []
        self.y_pred_test_proba_retrained_hist = []
        self.y_valid_train = y_valid_train
        self.y_add_valid_use = y_add_valid_use
        self.y_test = y_test

    @abstractmethod
    def prepare_resampling(self) -> None:
        """
        Prepare the resampling for the optimization.
        """
        pass

    def prepare_study(self) -> None:
        """
        Prepare the study for the optimization.
        """
        # if results_path does not exist, create it
        if not os.path.exists(self.results_path):
            try:
                os.makedirs(self.results_path)
            except FileExistsError:
                pass
        study_name = os.path.join(self.results_path, self.file_name)
        storage = f"sqlite:///{study_name}.db"
        # raise warning if study already exists
        if os.path.exists(f"{study_name}.db"):
            print(f"WARNING: Study {study_name} already exists. Overwriting it.")
            # remove the study
            os.remove(f"{study_name}.db")
        sampler = SmacSampler(
            seed=self.seed,
            configspace_search_space=self.classifier.get_configspace_search_space(
                n_train_samples=self.train_size
            ),
            internal_optuna_search_space=self.classifier.get_internal_optuna_search_space(
                n_train_samples=self.train_size
            ),
            n_trials=self.n_trials,
            fid=study_name,
        )
        self.study = optuna.create_study(
            storage=storage,
            sampler=sampler,
            study_name=study_name,
            direction="maximize",
        )
        self.study_name = study_name
        self.storage = storage

    @abstractmethod
    def objective(self, trial: Trial) -> float:
        """
        Objective function for the optimization.
        """
        pass

    @abstractmethod
    def store_results(self) -> None:
        """
        Store additional results.
        """
        pass

    def run(self) -> None:
        """
        Finish all preparations.
        Run the optimization.
        Stores the results in parquet along the study database.
        """
        self.prepare_data()

        self.prepare_resampling()

        self.prepare_study()

        # write a file with the parameters
        params = {
            "classifier": self.classifier.classifier_id,
            "data_id": self.data_id,
            "valid_type": self.valid_type,
            "train_valid_size": self.train_valid_size,
            "reshuffle": self.reshuffle,
            "n_splits": self.n_splits,
            "n_repeats": self.n_repeats,
            "valid_frac": self.valid_frac,
            "test_size": self.test_size,
            "n_trials": self.n_trials,
            "file_name": self.file_name,
            "seed": self.seed,
            "has_missings": self.has_missings,
            "labels": self.labels,
            "n_classes": self.n_classes,
            "multiclass": self.multiclass,
            "metric": self.metric,
            "metrics": self.metrics,
            "metrics_direction": self.metrics_direction,
            "cv_metric_to_metric": self.cv_metric_to_metric,
            "results_path": self.results_path,
            "study_name": self.study_name,
            "storage": self.storage,
        }
        with open(
            os.path.join(self.results_path, f"{self.file_name}_params.json"), "w"
        ) as f:
            json.dump(params, f, indent=4)

        np.random.seed(self.seed)
        random.seed(self.seed)
        self.study.optimize(self.objective, n_trials=self.n_trials)

        self.store_results()
