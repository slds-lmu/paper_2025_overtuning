import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class PostSelector(ABC):
    """
    Abstract class for a post selector.
    """

    def __init__(
        self,
        id: str,
        result_analyzer: "ResultAnalyzer",
        supported_valid_types: list = [
            "holdout",
            "cv",
            "cv_repeated",
            "repeatedholdout",
        ],
        supported_reshufflings: list = [True, False],
        resolution_sparse: bool = False,
        additional_iterationwise_results: Optional[List[str]] = None,
        bootstrap_results_path: str = os.path.abspath("../bootstrap_results"),
    ):
        self.id = id
        self.result_analyzer = result_analyzer
        self.supported_valid_types = supported_valid_types
        self.supported_reshufflings = supported_reshufflings
        self.resolution_sparse = resolution_sparse
        self.additional_iterationwise_results = additional_iterationwise_results
        self.bootstrap_results_path = bootstrap_results_path

    def select(self, iteration: int, metric: str, **kwargs) -> int:
        """
        Function to select a configuration.
        Performs some checks and then calls the _select function.
        """
        if self.result_analyzer.valid_type not in self.supported_valid_types:
            raise ValueError(
                f"Valid type {self.result_analyzer.valid_type} not supported by post selector {self.id}"
            )
        if self.result_analyzer.reshuffle not in self.supported_reshufflings:
            raise ValueError(
                f"Reshuffling {self.result_analyzer.reshuffle} not supported by post selector {self.id}"
            )

        return self._select(iteration=iteration, metric=metric, **kwargs)

    @abstractmethod
    def _select(self, iteration: int, metric: str, **kwargs) -> int:
        """
        Function to select a configuration.
        """
        pass

    def reset(self):
        """
        Reset the post selector.
        Calls the _reset function.
        """
        self._reset()

    @abstractmethod
    def _reset(self):
        """
        Reset the post selector.
        """
        pass


class PostSelectorNaive(PostSelector):
    """
    Selects the configuration with the lowest validation score.
    """

    def __init__(self, result_analyzer: "ResultAnalyzer"):
        super().__init__(id="naive", result_analyzer=result_analyzer)

    def _select(self, iteration: int, metric: str, **kwargs) -> int:
        """
        Selects the configuration with the lowest validation score.
        """
        valid = self.result_analyzer.results_raw[metric]["valid"].values[
            : iteration + 1
        ]
        selected = np.argmin(valid)
        return selected

    def _reset(self):
        """
        Reset the post selector.
        """
        pass
