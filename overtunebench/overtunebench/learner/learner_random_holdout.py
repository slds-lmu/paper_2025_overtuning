import os

from optuna.trial import Trial
from sklearn.model_selection import train_test_split

from overtunebench.algorithms.classifier import Classifier
from overtunebench.learner.learner_random import LearnerRandom
from overtunebench.metrics import compute_metric
from overtunebench.utils import (
    bootstrap_test_performance,
    construct_x_and_y_add_valid,
    save_list_of_1d_arrays,
    save_list_of_pd_arrays,
    save_single_array,
)


class LearnerRandomHoldout(LearnerRandom):
    def __init__(
        self,
        classifier: Classifier,
        data_id: int,
        train_valid_size: int,
        reshuffle: bool,
        valid_frac: float,
        test_size: int,
        add_valid_size: int,
        n_trials: int,
        seed: int,
    ):
        if reshuffle:
            results_path = os.path.abspath(
                os.path.join(
                    "results",
                    f"results_{classifier.classifier_id}_{data_id}_holdout{str(valid_frac).replace('.', '')}_reshuffle_{str(train_valid_size)}_{str(test_size)}",
                )
            )
        else:
            results_path = os.path.abspath(
                os.path.join(
                    "results",
                    f"results_{classifier.classifier_id}_{data_id}_holdout{str(valid_frac).replace('.', '')}_{str(train_valid_size)}_{str(test_size)}",
                )
            )
        super().__init__(
            classifier=classifier,
            data_id=data_id,
            valid_type="holdout",
            train_valid_size=train_valid_size,
            reshuffle=reshuffle,
            valid_frac=valid_frac,
            n_splits=None,
            n_repeats=None,
            test_size=test_size,
            add_valid_size=add_valid_size,
            n_trials=n_trials,
            seed=seed,
            results_path=results_path,
            cv_metric_to_metric=None,
        )

    def prepare_resampling(self) -> None:
        """
        Prepare the resampling for the optimization.
        """
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            self.x_valid_train,
            self.y_valid_train,
            test_size=self.valid_size,
            random_state=self.seed,
            stratify=self.y_valid_train,
        )
        self.train_size = self.x_train.shape[0]

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

                save_list_of_1d_arrays(
                    os.path.join(self.results_path, f"{self.file_name}_{file}.parquet"),
                    data=data,
                )

            # store y_valid_train, y_test
            for file in [
                # "y_valid_train",
                "y_test"
            ]:
                save_single_array(
                    os.path.join(self.results_path, f"{self.file_name}_{file}.parquet"),
                    data=getattr(self, file),
                )

            # store y_pred_train_proba_hist, y_pred_valid_proba_hist, y_pred_test_proba_hist, y_pred_valid_train_proba_hist, y_pred_test_proba_retrained_hist
            for file in [
                # "y_pred_train_proba_hist",
                "y_pred_valid_proba_hist",
                "y_pred_test_proba_hist",
                # "y_pred_valid_train_proba_hist",
                "y_pred_test_proba_retrained_hist",
            ]:
                save_list_of_pd_arrays(
                    os.path.join(self.results_path, f"{self.file_name}_{file}.parquet"),
                    data=getattr(self, file),
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

            save_list_of_1d_arrays(
                os.path.join(
                    self.results_path, f"{self.file_name}_y_add_valid_hist.parquet"
                ),
                data=data,
            )

            # store y_pred_add_valid_proba_hist
            save_list_of_pd_arrays(
                os.path.join(
                    self.results_path,
                    f"{self.file_name}_y_pred_add_valid_proba_hist.parquet",
                ),
                data=self.y_pred_add_valid_proba_hist,
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
            self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
                self.x_valid_train,
                self.y_valid_train,
                test_size=self.valid_size,
                random_state=self.seed + (trial.number * 500000),
                stratify=self.y_valid_train,
            )

        self.y_train_hist.append(self.y_train)
        self.y_valid_hist.append(self.y_valid)

        self.x_add_valid, self.y_add_valid = construct_x_and_y_add_valid(
            self.x_valid, self.y_valid, self.x_add_valid_use, self.y_add_valid_use
        )
        self.y_add_valid_hist.append(self.y_add_valid)

        # fit the classifier
        self.classifier.fit(
            trial=trial,
            x_train=self.x_train,
            y_train=self.y_train,
            x_valid=self.x_valid,
            y_valid=self.y_valid,
            cat_features=self.cat_features,
        )

        # predict on the train, valid, test and add_valid set and compute the metrics
        predictions = {}
        predictions_proba = {}
        metrics_train = {}
        metrics_valid = {}
        metrics_add_valid = {}
        metrics_test = {}

        for data in ["x_train", "x_valid", "x_add_valid", "x_test"]:
            predictions[data] = self.classifier.predict(getattr(self, data))
            predictions_proba[data] = self.classifier.predict_proba(getattr(self, data))
        for metric in self.metrics:
            metrics_train[metric] = compute_metric(
                self.y_train,
                y_pred=predictions["x_train"],
                y_pred_proba=predictions_proba["x_train"],
                metric=metric,
                labels=self.labels,
                multiclass=self.multiclass,
            )
            metrics_valid[metric] = compute_metric(
                self.y_valid,
                y_pred=predictions["x_valid"],
                y_pred_proba=predictions_proba["x_valid"],
                metric=metric,
                labels=self.labels,
                multiclass=self.multiclass,
            )
            metrics_add_valid[metric] = compute_metric(
                self.y_add_valid,
                y_pred=predictions["x_add_valid"],
                y_pred_proba=predictions_proba["x_add_valid"],
                metric=metric,
                labels=self.labels,
                multiclass=self.multiclass,
            )
            metrics_test[metric] = compute_metric(
                self.y_test,
                y_pred=predictions["x_test"],
                y_pred_proba=predictions_proba["x_test"],
                metric=metric,
                labels=self.labels,
                multiclass=self.multiclass,
            )
            trial.set_user_attr(f"{metric}_train", metrics_train[metric])
            trial.set_user_attr(f"{metric}_valid", metrics_valid[metric])
            trial.set_user_attr(f"{metric}_add_valid", metrics_add_valid[metric])
            trial.set_user_attr(f"{metric}_test", metrics_test[metric])

        if self.bootstrap_test:
            # bootstrap the test performance
            for metric in self.metrics:
                metric_test_bootstrap = bootstrap_test_performance(
                    y_test=self.y_test,
                    y_pred=predictions["x_test"],
                    y_pred_proba=predictions_proba["x_test"],
                    metric=metric,
                    labels=self.labels,
                    multiclass=self.multiclass,
                    seed=self.seed,
                )
                trial.set_user_attr(f"{metric}_test_bootstrap", metric_test_bootstrap)
                average_metric_test_bootstrap = sum(metric_test_bootstrap) / len(
                    metric_test_bootstrap
                )
                trial.set_user_attr(
                    f"{metric}_test_bootstrap_average", average_metric_test_bootstrap
                )

        self.y_pred_train_proba_hist.append(predictions_proba["x_train"])
        self.y_pred_valid_proba_hist.append(predictions_proba["x_valid"])
        self.y_pred_add_valid_proba_hist.append(predictions_proba["x_add_valid"])
        self.y_pred_test_proba_hist.append(predictions_proba["x_test"])

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

        # return the validation accuracy
        return metrics_valid["accuracy"]
