import json
import os

import numpy as np
from optuna.trial import Trial
from sklearn.model_selection import StratifiedShuffleSplit

from overtunebench.algorithms.classifier import Classifier
from overtunebench.learner.learner_random import LearnerRandom
from overtunebench.metrics import compute_metric
from overtunebench.utils import (
    NumpyArrayEncoder,
    bootstrap_test_performance,
    check_y_predict_proba,
    construct_x_and_y_add_valid,
    save_list_of_list_of_1d_arrays,
    save_list_of_list_of_pd_arrays,
    save_list_of_pd_arrays,
    save_single_array,
)


class LearnerRandomRepeatedHoldout(LearnerRandom):
    def __init__(
        self,
        classifier: Classifier,
        data_id: int,
        train_valid_size: int,
        reshuffle: bool,
        valid_frac: float,
        n_repeats: int,
        test_size: int,
        add_valid_size: int,
        n_trials: int,
        seed: int,
    ):
        # Note: add_valid size is the size of the total additional validation set, not the size of the additional validation set per repeat
        #       this is somewhat in contrast to LearnerRandomHoldout, where add_valid_size is the size of the additional validation set
        if reshuffle:
            results_path = os.path.abspath(
                os.path.join(
                    "results",
                    f"results_{classifier.classifier_id}_{data_id}_repeatedholdout{str(valid_frac).replace('.', '')}r{n_repeats}_reshuffle_{train_valid_size}_{test_size}",
                )
            )
        else:
            results_path = os.path.abspath(
                os.path.join(
                    "results",
                    f"results_{classifier.classifier_id}_{data_id}_repeatedholdout{str(valid_frac).replace('.', '')}r{n_repeats}_{train_valid_size}_{test_size}",
                )
            )

        cv_metric_to_metric = {
            "accuracies": "accuracy",
            "balanced_accuracies": "balanced_accuracy",
            "loglosses": "logloss",
            "aucs": "auc",
        }

        super().__init__(
            classifier=classifier,
            data_id=data_id,
            valid_type="repeatedholdout",
            train_valid_size=train_valid_size,
            reshuffle=reshuffle,
            valid_frac=valid_frac,
            n_splits=None,
            n_repeats=n_repeats,
            test_size=test_size,
            add_valid_size=add_valid_size,
            n_trials=n_trials,
            seed=seed,
            results_path=results_path,
            cv_metric_to_metric=cv_metric_to_metric,
        )

    def prepare_resampling(self) -> None:
        """
        Prepare the resampling for the optimization.
        """
        self.cv = StratifiedShuffleSplit(
            n_splits=self.n_repeats, test_size=self.valid_size, random_state=self.seed
        )
        self.cv_splits = list(self.cv.split(self.x_valid_train, self.y_valid_train))
        self.train_size = self.y_valid_train.shape[0] - self.valid_size
        self.cv_splits_hist_train = []
        self.cv_splits_hist_valid = []

        # split add_valid_use data and repeat it n_repeats times
        self.cv_add_valid = StratifiedShuffleSplit(
            n_splits=self.n_repeats,
            test_size=int(self.add_valid_size / self.n_repeats),
            random_state=self.seed,
        )
        self.cv_splits_add_valid = list(
            self.cv_add_valid.split(self.x_add_valid_use, self.y_add_valid_use)
        )
        self.cv_splits_add_valid_hist_valid = []

    def store_results(self) -> None:
        """
        Store additional results.
        """
        if self.write_results_to_disk:
            # store y_train_hist, y_valid_hist
            for file in [
                # "y_train_hist",
                "y_valid_hist"
            ]:
                data = getattr(self, file)
                # if we do not reshuffle, then y_train_hist and y_valid_hist are the same for all trials, so we only store the first one
                if not self.reshuffle:
                    data = [data[0]]

                save_list_of_list_of_1d_arrays(
                    os.path.join(self.results_path, f"{self.file_name}_{file}.parquet"),
                    data=data,
                )

            # store y_valid_train, y_test
            for file in ["y_valid_train", "y_test"]:
                save_single_array(
                    os.path.join(self.results_path, f"{self.file_name}_{file}.parquet"),
                    data=getattr(self, file),
                )

            # store y_pred_train_proba_hist, y_pred_valid_proba_hist, y_pred_test_proba_hist, y_pred_valid_train_proba_hist, y_pred_test_proba_retrained_hist
            for file in [
                # "y_pred_train_proba_hist",
                "y_pred_valid_proba_hist",
                "y_pred_test_proba_hist",
            ]:
                save_list_of_list_of_pd_arrays(
                    os.path.join(self.results_path, f"{self.file_name}_{file}.parquet"),
                    data=getattr(self, file),
                )

            # store y_pred_valid_train_proba_hist, y_pred_test_proba_retrained_hist
            for file in [
                # "y_pred_valid_train_proba_hist",
                "y_pred_test_proba_retrained_hist",
            ]:
                save_list_of_pd_arrays(
                    os.path.join(self.results_path, f"{self.file_name}_{file}.parquet"),
                    data=getattr(self, file),
                )

            # store cv_splits_hist_train and cv_splits_hist_valid
            for file in [
                # "cv_splits_hist_train",
                "cv_splits_hist_valid"
            ]:
                data = getattr(self, file)
                # if we do not reshuffle, then cv_splits_hist_train and cv_splits_hist_valid are the same for all trials, so we only store the first one
                if not self.reshuffle:
                    data = [data[0]]

                save_list_of_list_of_1d_arrays(
                    os.path.join(self.results_path, f"{self.file_name}_{file}.parquet"),
                    data=data,
                )

            # store y_add_valid_use
            save_single_array(
                os.path.join(
                    self.results_path, f"{self.file_name}_y_add_valid_use.parquet"
                ),
                data=self.y_add_valid_use,
            )

            # store y_add_valid_hist
            data = self.y_add_valid_hist
            # if we do not reshuffle, then y_add_valid_hist is the same for all trials, so we only store the first one
            if not self.reshuffle:
                data = [data[0]]

            save_list_of_list_of_1d_arrays(
                os.path.join(
                    self.results_path, f"{self.file_name}_y_add_valid_hist.parquet"
                ),
                data=data,
            )

            # store y_pred_add_valid_proba_hist
            save_list_of_list_of_pd_arrays(
                os.path.join(
                    self.results_path,
                    f"{self.file_name}_y_pred_add_valid_proba_hist.parquet",
                ),
                data=self.y_pred_add_valid_proba_hist,
            )

            # store cv_splits_add_valid_hist_valid
            data = self.cv_splits_add_valid_hist_valid
            # if we do not reshuffle, then cv_splits_add_valid_hist_valid is the same for all trials, so we only store the first one
            if not self.reshuffle:
                data = [data[0]]

            save_list_of_list_of_1d_arrays(
                os.path.join(
                    self.results_path,
                    f"{self.file_name}_cv_splits_add_valid_hist_valid.parquet",
                ),
                data=data,
            )

    def objective(self, trial: Trial) -> float:
        """
        Objective function for the optimization.
        """
        # construct classifier pipeline
        self.classifier.construct_pipeline(
            trial,
            refit=False,
            cat_features=self.cat_features,
            num_features=self.num_features,
            n_train_samples=self.train_size,
        )

        if self.reshuffle:
            self.cv = StratifiedShuffleSplit(
                n_splits=self.n_repeats,
                test_size=self.valid_size,
                random_state=self.seed + (trial.number * 500000),
            )
            self.cv_splits = list(self.cv.split(self.x_valid_train, self.y_valid_train))

            # split add_valid_use data and repeat it n_repeats times
            self.cv_add_valid = StratifiedShuffleSplit(
                n_splits=self.n_repeats,
                test_size=int(self.add_valid_size / self.n_repeats),
                random_state=self.seed + (trial.number * 500000),
            )
            self.cv_splits_add_valid = list(
                self.cv_add_valid.split(self.x_add_valid_use, self.y_add_valid_use)
            )

        cv_splits_hist_train_tmp = []
        cv_splits_hist_valid_tmp = []
        cv_splits_add_valid_hist_valid_tmp = []
        y_train_hist_tmp = []
        y_valid_hist_tmp = []
        y_add_valid_hist_tmp = []

        # for each repeat fit the classifier and predict on the train, valid, test and add_valid set and compute the metrics
        predictions = dict(
            [(data, []) for data in ["x_train", "x_valid", "x_add_valid", "x_test"]]
        )
        predictions_proba = dict(
            [(data, []) for data in ["x_train", "x_valid", "x_add_valid", "x_test"]]
        )
        cv_metrics_train = dict([(metric, []) for metric in self.metrics])
        cv_metrics_valid = dict([(metric, []) for metric in self.metrics])
        cv_metrics_add_valid = dict([(metric, []) for metric in self.metrics])
        cv_metrics_test = dict([(metric, []) for metric in self.metrics])

        for repeat in range(self.n_repeats):
            train_index, valid_index = self.cv_splits[repeat]
            cv_splits_hist_train_tmp.append(train_index)
            cv_splits_hist_valid_tmp.append(valid_index)
            x_train = self.x_valid_train.iloc[train_index]
            x_valid = self.x_valid_train.iloc[valid_index]
            y_train = self.y_valid_train[train_index]
            y_valid = self.y_valid_train[valid_index]
            y_train_hist_tmp.append(y_train)
            y_valid_hist_tmp.append(y_valid)

            add_valid_index = self.cv_splits_add_valid[repeat][1]
            cv_splits_add_valid_hist_valid_tmp.append(
                np.concatenate([valid_index, add_valid_index])
            )
            x_add_valid, y_add_valid = construct_x_and_y_add_valid(
                x_valid=x_valid,
                y_valid=y_valid,
                x_add_valid=self.x_add_valid_use.iloc[add_valid_index],
                y_add_valid=self.y_add_valid_use[add_valid_index],
            )
            y_add_valid_hist_tmp.append(y_add_valid)

            self.classifier.fit(
                trial=trial,
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                cat_features=self.cat_features,
            )
            for data in ["x_train", "x_valid", "x_add_valid", "x_test"]:
                if data == "x_test":
                    predictions[data].append(
                        self.classifier.predict(getattr(self, data))
                    )
                    predictions_proba[data].append(
                        self.classifier.predict_proba(getattr(self, data))
                    )
                else:
                    predictions[data].append(self.classifier.predict(eval(data)))
                    predictions_proba[data].append(
                        self.classifier.predict_proba(eval(data))
                    )
            for metric in self.metrics:
                cv_metrics_train[metric].append(
                    compute_metric(
                        y_train,
                        y_pred=predictions["x_train"][-1],
                        y_pred_proba=predictions_proba["x_train"][-1],
                        metric=metric,
                        labels=self.labels,
                        multiclass=self.multiclass,
                    )
                )
                cv_metrics_valid[metric].append(
                    compute_metric(
                        y_valid,
                        y_pred=predictions["x_valid"][-1],
                        y_pred_proba=predictions_proba["x_valid"][-1],
                        metric=metric,
                        labels=self.labels,
                        multiclass=self.multiclass,
                    )
                )
                cv_metrics_add_valid[metric].append(
                    compute_metric(
                        y_add_valid,
                        y_pred=predictions["x_add_valid"][-1],
                        y_pred_proba=predictions_proba["x_add_valid"][-1],
                        metric=metric,
                        labels=self.labels,
                        multiclass=self.multiclass,
                    )
                )
                cv_metrics_test[metric].append(
                    compute_metric(
                        self.y_test,
                        y_pred=predictions["x_test"][-1],
                        y_pred_proba=predictions_proba["x_test"][-1],
                        metric=metric,
                        labels=self.labels,
                        multiclass=self.multiclass,
                    )
                )

        self.cv_splits_hist_train.append(cv_splits_hist_train_tmp)
        self.cv_splits_hist_valid.append(cv_splits_hist_valid_tmp)
        self.cv_splits_add_valid_hist_valid.append(cv_splits_add_valid_hist_valid_tmp)
        self.y_train_hist.append(y_train_hist_tmp)
        self.y_valid_hist.append(y_valid_hist_tmp)
        self.y_add_valid_hist.append(y_add_valid_hist_tmp)
        self.y_pred_train_proba_hist.append(predictions_proba["x_train"])
        self.y_pred_valid_proba_hist.append(predictions_proba["x_valid"])
        self.y_pred_add_valid_proba_hist.append(predictions_proba["x_add_valid"])
        self.y_pred_test_proba_hist.append(predictions_proba["x_test"])

        for cv_metric in self.cv_metric_to_metric.keys():
            metric = self.cv_metric_to_metric[cv_metric]
            trial.set_user_attr(
                f"{cv_metric}_train",
                json.dumps(cv_metrics_train[metric], cls=NumpyArrayEncoder),
            )
            trial.set_user_attr(
                f"{cv_metric}_valid",
                json.dumps(cv_metrics_valid[metric], cls=NumpyArrayEncoder),
            )
            trial.set_user_attr(
                f"{cv_metric}_add_valid",
                json.dumps(cv_metrics_add_valid[metric], cls=NumpyArrayEncoder),
            )
            trial.set_user_attr(
                f"{cv_metric}_test",
                json.dumps(cv_metrics_test[metric], cls=NumpyArrayEncoder),
            )

        # compute the mean of the metrics over the repeats
        metrics_train = {}
        metrics_valid = {}
        metrics_add_valid = {}
        metrics_test = {}

        for metric in self.metrics:
            metrics_train[metric] = np.mean(cv_metrics_train[metric])
            metrics_valid[metric] = np.mean(cv_metrics_valid[metric])
            metrics_add_valid[metric] = np.mean(cv_metrics_add_valid[metric])
            metrics_test[metric] = np.mean(cv_metrics_test[metric])
            trial.set_user_attr(f"{metric}_train", metrics_train[metric])
            trial.set_user_attr(f"{metric}_valid", metrics_valid[metric])
            trial.set_user_attr(f"{metric}_add_valid", metrics_add_valid[metric])
            trial.set_user_attr(f"{metric}_test", metrics_test[metric])

        # compute the metrics on the test set also in ensemble style
        metrics_test_ensemble = {}
        predictions_proba_test_ensemble = np.mean(predictions_proba["x_test"], axis=0)
        row_sums = predictions_proba_test_ensemble.sum(axis=1, keepdims=True)
        predictions_proba_test_ensemble = predictions_proba_test_ensemble / row_sums
        check_y_predict_proba(predictions_proba_test_ensemble)
        predictions_test_ensemble = np.argmax(predictions_proba_test_ensemble, axis=1)
        for metric in self.metrics:
            metrics_test_ensemble[metric] = compute_metric(
                self.y_test,
                y_pred=predictions_test_ensemble,
                y_pred_proba=predictions_proba_test_ensemble,
                metric=metric,
                labels=self.labels,
                multiclass=self.multiclass,
            )
            trial.set_user_attr(
                f"{metric}_test_ensemble", metrics_test_ensemble[metric]
            )

        if self.bootstrap_test:
            # bootstrap the ensemble style test performance
            for metric in self.metrics:
                metric_test_bootstrap = bootstrap_test_performance(
                    y_test=self.y_test,
                    y_pred=predictions_test_ensemble,
                    y_pred_proba=predictions_proba_test_ensemble,
                    metric=metric,
                    labels=self.labels,
                    multiclass=self.multiclass,
                    seed=self.seed,
                )
                trial.set_user_attr(
                    f"{metric}_test_ensemble_bootstrap", metric_test_bootstrap
                )
                average_metric_test_bootstrap = sum(metric_test_bootstrap) / len(
                    metric_test_bootstrap
                )
                trial.set_user_attr(
                    f"{metric}_test_ensemble_bootstrap_average",
                    average_metric_test_bootstrap,
                )

        # refit on the train_valid set
        self.classifier.construct_pipeline(
            trial,
            refit=True,
            cat_features=self.cat_features,
            num_features=self.num_features,
        )
        self.classifier.fit(
            trial=trial,
            x_train=self.x_valid_train,
            y_train=self.y_valid_train,
            cat_features=self.cat_features,
        )

        # predict on the train_valid set and compute the metrics
        predictions["x_valid_train"] = self.classifier.predict(self.x_valid_train)
        predictions_proba["x_valid_train"] = self.classifier.predict_proba(
            self.x_valid_train
        )
        metrics_valid_train = {}
        for metric in self.metrics:
            metrics_valid_train[metric] = compute_metric(
                self.y_valid_train,
                y_pred=predictions["x_valid_train"],
                y_pred_proba=predictions_proba["x_valid_train"],
                metric=metric,
                labels=self.labels,
                multiclass=self.multiclass,
            )
            trial.set_user_attr(f"{metric}_valid_train", metrics_valid_train[metric])

        self.y_pred_valid_train_proba_hist.append(predictions_proba["x_valid_train"])

        # predict on the test set and compute the metrics
        predictions["x_test_retrained"] = self.classifier.predict(self.x_test)
        predictions_proba["x_test_retrained"] = self.classifier.predict_proba(
            self.x_test
        )
        metrics_test_retrained = {}
        for metric in self.metrics:
            metrics_test_retrained[metric] = compute_metric(
                self.y_test,
                y_pred=predictions["x_test_retrained"],
                y_pred_proba=predictions_proba["x_test_retrained"],
                metric=metric,
                labels=self.labels,
                multiclass=self.multiclass,
            )
            trial.set_user_attr(
                f"{metric}_test_retrained", metrics_test_retrained[metric]
            )

        if self.bootstrap_test:
            # bootstrap the retrained test performance
            for metric in self.metrics:
                metric_test_retrained_bootstrap = bootstrap_test_performance(
                    y_test=self.y_test,
                    y_pred=predictions["x_test_retrained"],
                    y_pred_proba=predictions_proba["x_test_retrained"],
                    metric=metric,
                    labels=self.labels,
                    multiclass=self.multiclass,
                    seed=self.seed,
                )
                trial.set_user_attr(
                    f"{metric}_test_retrained_bootstrap",
                    metric_test_retrained_bootstrap,
                )
                average_metric_test_retrained_bootstrap = sum(
                    metric_test_retrained_bootstrap
                ) / len(metric_test_retrained_bootstrap)
                trial.set_user_attr(
                    f"{metric}_test_retrained_bootstrap_average",
                    average_metric_test_retrained_bootstrap,
                )

        self.y_pred_test_proba_retrained_hist.append(
            predictions_proba["x_test_retrained"]
        )

        self.classifier.reset()

        # return the validation accuracy (mean over repeats)
        return metrics_valid["accuracy"]
