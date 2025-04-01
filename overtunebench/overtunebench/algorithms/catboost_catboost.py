import json
from typing import List, Optional

import numpy as np
from catboost import CatBoostClassifier
from ConfigSpace import ConfigurationSpace, Float, Integer
from hebo.design_space import DesignSpace
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import Trial

from overtunebench.algorithms.classifier import Classifier
from overtunebench.utils import NumpyArrayEncoder


# search space from https://github.com/naszilla/tabzilla/blob/d689be5603f0a5fc8de30758d68b7122bcf46719/TabZilla/models/tree_models.py#L112
class CatBoost(Classifier):
    def __init__(self, seed: int, default: bool = False):
        if default:
            classifier_id = "catboost_default"
        else:
            classifier_id = "catboost"

        super().__init__(
            classifier_id=classifier_id,
            impute_x_cat=False,
            impute_x_num=False,
            encode_x=False,
            scale_x=False,
            seed=seed,
            default=default,
        )
        self.learning_rate = None
        self.depth = None
        self.l2_leaf_reg = None
        self.actual_iterations = []
        self.in_refit_mode = False

    def get_hebo_search_space(self, **kwargs):
        """
        Get the HEBO search space.
        """
        hebo_params = [
            {
                "name": "learning_rate",
                "type": "pow",
                "lb": 0.01,
                "ub": 0.3,
                "base": np.e,
            },
            {"name": "depth", "type": "pow_int", "lb": 2, "ub": 12, "base": np.e},
            {"name": "l2_leaf_reg", "type": "pow", "lb": 0.5, "ub": 30, "base": np.e},
        ]
        hebo_search_space = DesignSpace().parse(hebo_params)
        return hebo_search_space

    def get_configspace_search_space(self, **kwargs):
        """
        Get the configspace search space.
        """
        cs = ConfigurationSpace(seed=self.seed)
        learning_rate = Float(name="learning_rate", bounds=[0.01, 0.3], log=True)
        depth = Integer(name="depth", bounds=[2, 12], log=True)
        l2_leaf_reg = Float(name="l2_leaf_reg", bounds=[0.5, 30], log=True)
        cs.add([learning_rate, depth, l2_leaf_reg])
        return cs

    def get_internal_optuna_search_space(self, **kwargs):
        """
        Get the internal Optuna search space.
        """
        internal_optuna_search_space = {
            "learning_rate": FloatDistribution(low=0.01, high=0.3, step=None, log=True),
            "depth": IntDistribution(low=2, high=12, step=1, log=True),
            "l2_leaf_reg": FloatDistribution(low=0.5, high=30, step=None, log=True),
        }
        return internal_optuna_search_space

    def construct_classifier(self, trial: Trial, **kwargs) -> None:
        """
        Construct the classifier based on the trial.
        """
        if self.default:
            self.learning_rate = 0.03
            self.depth = 6
            self.l2_leaf_reg = 3
        else:
            self.learning_rate = trial.suggest_float(
                "learning_rate", low=0.01, high=0.3, log=True
            )
            self.depth = trial.suggest_int("depth", low=2, high=12, log=True)
            self.l2_leaf_reg = trial.suggest_float(
                "l2_leaf_reg", low=0.5, high=30, log=True
            )

        classifier = CatBoostClassifier(
            iterations=2000,
            od_type="Iter",
            od_wait=20,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
            # thread_count = 16 for some catboost runs with hebo, thread_count was set to 16 and HPO was run on a whole node with 16 threads, 128GB RAM to stay below 7 days of runtime
        )

        self.classifier = classifier
        self.in_refit_mode = False

    def update_actual_iterations(self) -> None:
        """
        Update the actual iterations used for refitting based on the best iteration(s).
        """
        if self.classifier is None:
            raise ValueError("Classifier is None")

        self.actual_iterations.append(self.classifier.get_best_iteration())

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

        classifier = CatBoostClassifier(
            iterations=actual_iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
            # thread_count = 16 for some catboost runs with hebo, thread_count was set to 16 and HPO was run on a whole node with 16 threads, 128GB RAM to stay below 7 days of runtime
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
        # check that if cat_features is not None and not empty, then the respective columns are of dtype object
        if cat_features is not None and len(cat_features) > 0:
            if x_train[:, cat_features].dtype != np.dtype("object"):
                raise ValueError(
                    "If cat_features is not None and not empty, then the respective columns of x_train must be of dtype object"
                )
            if x_valid is not None:
                if x_valid[:, cat_features].dtype != np.dtype("object"):
                    raise ValueError(
                        "If cat_features is not None and not empty, then the respective columns of x_valid must be of dtype object"
                    )
        if self.in_refit_mode:
            trial.set_user_attr(
                "actual_iterations",
                json.dumps(self.actual_iterations, cls=NumpyArrayEncoder),
            )
            self.classifier.fit(X=x_train, y=y_train, cat_features=cat_features)
        else:
            self.classifier.fit(
                X=x_train,
                y=y_train,
                eval_set=(x_valid, y_valid),
                cat_features=cat_features,
            )
            self.update_actual_iterations()

    def reset(self) -> None:
        """
        Reset the classifier.
        """
        super().reset()
        self.learning_rate = None
        self.depth = None
        self.l2_leaf_reg = None
        self.actual_iterations = []
        self.in_refit_mode = False
