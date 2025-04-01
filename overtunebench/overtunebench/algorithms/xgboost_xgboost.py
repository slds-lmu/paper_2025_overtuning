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


# search space from https://github.com/naszilla/tabzilla/blob/d689be5603f0a5fc8de30758d68b7122bcf46719/TabZilla/models/tree_models.py#L20
class XGBoost(Classifier):
    def __init__(self, seed: int, default: bool = False):
        if default:
            classifier_id = "xgboost_default"
        else:
            classifier_id = "xgboost"
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
        self.actual_iterations = []
        self.in_refit_mode = False

    def get_hebo_search_space(self, **kwargs):
        """
        Get the HEBO search space.
        """
        hebo_params = [
            {
                "name": "max_depth",
                "type": "pow_int",
                "lb": 2,
                "ub": 12,
                "base": np.e,
            },
            {"name": "alpha", "type": "pow", "lb": 1e-8, "ub": 1.0, "base": np.e},
            {"name": "lambda", "type": "pow", "lb": 1e-8, "ub": 1.0, "base": np.e},
            {"name": "eta", "type": "pow", "lb": 0.01, "ub": 0.3, "base": np.e},
        ]
        hebo_search_space = DesignSpace().parse(hebo_params)
        return hebo_search_space

    def get_configspace_search_space(self, **kwargs):
        """
        Get the configspace search space.
        """
        cs = ConfigurationSpace(seed=self.seed)
        max_depth = Integer(name="max_depth", bounds=[2, 12], log=True)
        alpha = Float(name="alpha", bounds=[1e-8, 1.0], log=True)
        lambda_ = Float(name="lambda", bounds=[1e-8, 1.0], log=True)
        eta = Float(name="eta", bounds=[0.01, 0.3], log=True)
        cs.add([max_depth, alpha, lambda_, eta])
        return cs

    def get_internal_optuna_search_space(self, **kwargs):
        """
        Get the internal Optuna search space.
        """
        internal_optuna_search_space = {
            "max_depth": IntDistribution(low=2, high=12, step=1, log=True),
            "alpha": FloatDistribution(low=1e-8, high=1.0, step=None, log=True),
            "lambda": FloatDistribution(low=1e-8, high=1.0, step=None, log=True),
            "eta": FloatDistribution(low=0.01, high=0.3, step=None, log=True),
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
        else:
            self.max_depth = trial.suggest_int("max_depth", low=2, high=12, log=True)
            self.alpha = trial.suggest_float("alpha", low=1e-8, high=1.0, log=True)
            self.lambda_ = trial.suggest_float("lambda", low=1e-8, high=1.0, log=True)
            self.eta = trial.suggest_float("eta", low=0.01, high=0.3, log=True)

        classifier = XGBClassifier(
            n_estimators=2000,
            early_stopping_rounds=20,
            max_depth=self.max_depth,
            reg_alpha=self.alpha,
            reg_lambda=self.lambda_,
            learning_rate=self.eta,
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
