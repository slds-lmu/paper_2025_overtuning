from typing import List, Optional

import numpy as np
from optuna.trial import Trial

from overtunebench.algorithms.classifier import Classifier


class Featureless(Classifier):
    def __init__(self, seed: int):
        super().__init__(
            classifier_id="featureless",
            impute_x_cat=False,
            impute_x_num=False,
            encode_x=False,
            scale_x=False,
            seed=seed,
        )

    def get_hebo_search_space(self, **kwargs):
        """
        Get the HEBO search space.
        """
        return None

    def get_configspace_search_space(self, **kwargs):
        """
        Get the configspace search space.
        """
        return None

    def get_internal_optuna_search_space(self, **kwargs):
        """
        Get the internal Optuna search space.
        """
        return None

    def construct_classifier(self, trial: Trial, **kwargs) -> None:
        """
        Construct the classifier based on the trial.
        """
        classifier = None

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
        raise ValueError("No classifier to fit, will trigger fallback")
