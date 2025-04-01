from typing import List, Optional

import numpy as np
from ConfigSpace import ConfigurationSpace, Float
from hebo.design_space import DesignSpace
from optuna.distributions import FloatDistribution
from optuna.trial import Trial
from sklearn.linear_model import LogisticRegression

from overtunebench.algorithms.classifier import Classifier


class LogReg(Classifier):
    def __init__(self, seed: int, default: bool = False):
        if default:
            classifier_id = "logreg_default"
        else:
            classifier_id = "logreg"
        super().__init__(
            classifier_id=classifier_id,
            impute_x_cat=True,
            impute_x_num=True,
            encode_x=True,
            scale_x=True,
            seed=seed,
            default=default,
        )

    def get_hebo_search_space(self, **kwargs):
        """
        Get the HEBO search space.
        """
        hebo_params = [
            {"name": "C", "type": "pow", "lb": 1e-6, "ub": 1e4, "base": np.e},
            {"name": "l1_ratio", "type": "num", "lb": 0.0, "ub": 1.0},
        ]
        hebo_search_space = DesignSpace().parse(hebo_params)
        return hebo_search_space

    def get_configspace_search_space(self, **kwargs):
        """
        Get the configspace search space.
        """
        cs = ConfigurationSpace(seed=self.seed)
        C = Float(name="C", bounds=[1e-6, 1e4], log=True)
        l1_ratio = Float(name="l1_ratio", bounds=[0.0, 1.0])
        cs.add([C, l1_ratio])
        return cs

    def get_internal_optuna_search_space(self, **kwargs):
        """
        Get the internal Optuna search space.
        """
        internal_optuna_search_space = {
            "C": FloatDistribution(low=1e-6, high=1e4, step=None, log=True),
            "l1_ratio": FloatDistribution(low=0.0, high=1.0, step=None),
        }
        return internal_optuna_search_space

    def construct_classifier(self, trial: Trial, **kwargs) -> None:
        """
        Construct the classifier based on the trial.
        """
        if self.default:
            c = 1
            l1_ratio = 0
        else:
            c = trial.suggest_float("C", low=1e-6, high=1e4, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", low=0.0, high=1.0)

        classifier = LogisticRegression(
            penalty="elasticnet",
            C=c,
            l1_ratio=l1_ratio,
            solver="saga",
            max_iter=1000,
        )

        self.classifier = classifier

    def construct_classifier_refit(self, trial: Trial, **kwargs) -> None:
        """
        Construct the classifier for refitting.
        Nothing done here.
        """
        pass

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
        Fit the classifier.
        """
        self.classifier.fit(x_train, y_train)
