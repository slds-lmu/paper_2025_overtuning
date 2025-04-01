from typing import List, Optional

import numpy as np
import torch
from optuna.trial import Trial
from tabpfn import TabPFNClassifier

from overtunebench.algorithms.classifier import Classifier


class TabPFN(Classifier):
    def __init__(self, seed: int):
        super().__init__(
            classifier_id="tabpfn",
            impute_x_cat=True,
            impute_x_num=False,
            encode_x=True,
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
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        # seed explicitly set to ensure reproducibility of TabPFN
        classifier = TabPFNClassifier(
            device=device, N_ensemble_configurations=32, seed=self.seed
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
        self.classifier.fit(x_train, y_train, overwrite_warning=True)
