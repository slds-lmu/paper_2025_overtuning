import json
from typing import List, Optional

import numpy as np
from ConfigSpace import ConfigurationSpace, Float, Integer
from hebo.design_space import DesignSpace
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import Trial
from xgboost import XGBClassifier

from overtunebench.algorithms.classifier import Classifier
from overtunebench.utils import NumpyArrayEncoder


# search space from https://github.com/LeoGrin/tabular-benchmark/blob/main/src/configs/model_configs/xgb_config.py
class XGBoostLarge(Classifier):
    def __init__(self, seed: int, default: bool = False):
        if default:
            classifier_id = "xgboost_large_default"
        else:
            classifier_id = "xgboost_large"
        super().__init__(
            classifier_id=classifier_id,
            impute_x_cat=True,
            impute_x_num=False,
            encode_x=True,
            scale_x=False,
            seed=seed,
            default=default,
        )

        self.max_depth = None
        self.alpha = None
        self.lambda_ = None
        self.eta = None

        self.min_child_weight = None
        self.subsample = None
        self.colsample_bylevel = None
        self.colsample_bytree = None
        self.gamma = None

        self.actual_iterations = []
        self.in_refit_mode = False

    def get_hebo_search_space(self, **kwargs):
        """
        Get the HEBO search space.
        """
        hebo_params = [
            {
                "name": "max_depth",
                "type": "int",
                "lb": 1,
                "ub": 11,
            },
            {"name": "alpha", "type": "pow", "lb": 1e-8, "ub": 1e2, "base": np.e},
            {"name": "lambda", "type": "pow", "lb": 1, "ub": 4, "base": np.e},
            {"name": "eta", "type": "pow", "lb": 1e-5, "ub": 0.7, "base": np.e},
            {
                "name": "min_child_weight",
                "type": "pow_int",
                "lb": 1,
                "ub": 100,
                "base": np.e,
            },
            {"name": "subsample", "type": "num", "lb": 0.5, "ub": 1.0},
            {"name": "colsample_bylevel", "type": "num", "lb": 0.5, "ub": 1.0},
            {"name": "colsample_bytree", "type": "num", "lb": 0.5, "ub": 1.0},
            {"name": "gamma", "type": "pow", "lb": 1e-8, "ub": 7, "base": np.e},
        ]
        hebo_search_space = DesignSpace().parse(hebo_params)
        return hebo_search_space

    def get_configspace_search_space(self, **kwargs):
        """
        Get the configspace search space.
        """
        cs = ConfigurationSpace(seed=self.seed)
        max_depth = Integer(name="max_depth", bounds=[1, 11], log=True)
        alpha = Float(name="alpha", bounds=[1e-8, 1e2], log=True)
        lambda_ = Float(name="lambda", bounds=[1, 4], log=True)
        eta = Float(name="eta", bounds=[1e-5, 0.7], log=True)
        min_child_weight = Integer(name="min_child_weight", bounds=[1, 100], log=True)
        subsample = (Float(name="subsample", bounds=[0.5, 1.0]),)
        colsample_bylevel = (Float(name="colsample_bylevel", bounds=[0.5, 1.0]),)
        colsample_bytree = (Float(name="colsample_bytree", bounds=[0.5, 1.0]),)
        gamma = Float(name="gamma", bounds=[1e-8, 7], log=True)
        cs.add(
            [
                max_depth,
                alpha,
                lambda_,
                eta,
                min_child_weight,
                subsample,
                colsample_bylevel,
                colsample_bytree,
                gamma,
            ]
        )
        return cs

    def get_internal_optuna_search_space(self, **kwargs):
        """
        Get the internal Optuna search space.
        """
        internal_optuna_search_space = {
            "max_depth": IntDistribution(low=1, high=11, step=1),
            "alpha": FloatDistribution(low=1e-8, high=1e2, step=None, log=True),
            "lambda": FloatDistribution(low=1, high=4, step=None, log=True),
            "eta": FloatDistribution(low=1e-5, high=0.7, step=None, log=True),
            "min_child_weight": IntDistribution(low=1, high=100, step=1, log=True),
            "subsample": FloatDistribution(low=0.5, high=1.0, step=None),
            "colsample_bylevel": FloatDistribution(low=0.5, high=1.0, step=None),
            "colsample_bytree": FloatDistribution(low=0.5, high=1.0, step=None),
            "gamma": FloatDistribution(low=1e-8, high=7, step=None, log=True),
        }
        return internal_optuna_search_space

    def construct_classifier(self, trial: Trial, **kwargs) -> None:
        """
        Construct the classifier based on the trial.
        """
        if self.default:
            self.max_depth = 6
            self.alpha = 0.0
            self.lambda_ = 1.0
            self.eta = 0.3

            self.min_child_weight = 1
            self.subsample = 1.0
            self.colsample_bylevel = 1.0
            self.colsample_bytree = 1.0
            self.gamma = 0.0

        else:
            self.max_depth = trial.suggest_int("max_depth", low=1, high=11)
            self.alpha = trial.suggest_float("alpha", low=1e-8, high=1e2, log=True)
            self.lambda_ = trial.suggest_float("lambda", low=1, high=4, log=True)
            self.eta = trial.suggest_float("eta", low=1e-5, high=0.7, log=True)

            self.min_child_weight = trial.suggest_int(
                "min_child_weight", low=1, high=100, log=True
            )
            self.subsample = trial.suggest_float("subsample", low=0.5, high=1.0)
            self.colsample_bylevel = trial.suggest_float(
                "colsample_bylevel", low=0.5, high=1.0
            )
            self.colsample_bytree = trial.suggest_float(
                "colsample_bytree", low=0.5, high=1.0
            )
            self.gamma = trial.suggest_float("gamma", low=1e-8, high=7, log=True)

        classifier = XGBClassifier(
            n_estimators=2000,
            early_stopping_rounds=20,
            max_depth=self.max_depth,
            reg_alpha=self.alpha,
            reg_lambda=self.lambda_,
            learning_rate=self.eta,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            verbosity=0,
            n_jobs=1,
        )

        self.classifier = classifier
        self.in_refit_mode = False

    def update_actual_iterations(self) -> None:
        """
        Update the actual iterations used for refitting based on the best iteration(s).
        """
        if self.classifier is None:
            raise ValueError("Classifier is None")

        self.actual_iterations.append(self.classifier.best_iteration)

    def construct_classifier_refit(self, trial: Trial, **kwargs) -> None:
        """
        Construct the classifier for refitting.
        """
        if len(self.actual_iterations) == 0:
            raise ValueError("No actual iterations stored")

        if len(self.actual_iterations) == 1:
            actual_iterations = self.actual_iterations[0]
        else:
            actual_iterations = int(
                round(sum(self.actual_iterations) / len(self.actual_iterations))
            )

        if actual_iterations < 1:
            actual_iterations = 1

        classifier = XGBClassifier(
            n_estimators=actual_iterations,
            max_depth=self.max_depth,
            reg_alpha=self.alpha,
            reg_lambda=self.lambda_,
            learning_rate=self.eta,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            verbosity=0,
            n_jobs=1,
        )

        self.classifier = classifier
        self.in_refit_mode = True

    def _fit(
        self,
        trial: Trial,
        x_train: np.array,
        y_train: np.array,
        x_valid: Optional[np.array] = None,
        y_valid: Optional[np.array] = None,
        cat_features: Optional[List[int]] = None,
    ) -> None:
        """
        Train the classifier.
        If in refit mode no early stopping is used but the number of iterations are set to the average of the best iterations, see update_actual_iterations and construct_classifier_refit.
        """
        if self.in_refit_mode:
            trial.set_user_attr(
                "actual_iterations",
                json.dumps(self.actual_iterations, cls=NumpyArrayEncoder),
            )
            self.classifier.fit(X=x_train, y=y_train, verbose=False)
        else:
            self.classifier.fit(
                X=x_train,
                y=y_train,
                eval_set=[(x_valid, y_valid)],
                verbose=False,
            )
            self.update_actual_iterations()

    def reset(self) -> None:
        """
        Reset the classifier.
        """
        super().reset()
        self.max_depth = None
        self.alpha = None
        self.lambda_ = None
        self.eta = None
        self.actual_iterations = []
        self.in_refit_mode = False
