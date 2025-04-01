from typing import List, Optional

import numpy as np
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from hebo.design_space import DesignSpace
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import Trial
from sklearn.neural_network import MLPClassifier

from overtunebench.algorithms.classifier import Classifier


class FunnelMLP(Classifier):
    def __init__(self, seed: int, default: bool = False):
        if default:
            classifier_id = "funnel_mlp_default"
        else:
            classifier_id = "funnel_mlp"

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
        n_train_samples = kwargs["n_train_samples"]
        # maximum batch size is the largest power of 2 that is smaller than the number of training samples
        max_batch_size_log2 = int(np.log2(n_train_samples))
        hebo_params = [
            {"name": "num_layers", "type": "int", "lb": 1, "ub": 5},
            {
                "name": "max_units",
                "type": "cat",
                "categories": [2**i for i in range(6, 10)],
            },
            {
                "name": "learning_rate",
                "type": "pow",
                "lb": 1e-4,
                "ub": 1e-1,
                "base": np.e,
            },
            {
                "name": "batch_size",
                "type": "cat",
                "categories": [2**i for i in range(4, max_batch_size_log2 + 1)],
            },
            {"name": "momentum", "type": "num", "lb": 0.1, "ub": 0.99},
            {"name": "alpha", "type": "pow", "lb": 1e-6, "ub": 1e-1, "base": np.e},
        ]
        hebo_search_space = DesignSpace().parse(hebo_params)
        return hebo_search_space

    def get_configspace_search_space(self, **kwargs):
        """
        Get the configspace search space.
        """
        n_train_samples = kwargs["n_train_samples"]
        # maximum batch size is the largest power of 2 that is smaller than the number of training samples
        max_batch_size_log2 = int(np.log2(n_train_samples))
        cs = ConfigurationSpace(seed=self.seed)
        num_layers = Integer(name="num_layers", bounds=[1, 5])
        max_units = Categorical(name="max_units", items=[2**i for i in range(6, 10)])
        learning_rate = Float(name="learning_rate", bounds=[1e-4, 1e-1], log=True)
        batch_size = Categorical(
            name="batch_size", items=[2**i for i in range(4, max_batch_size_log2 + 1)]
        )
        momentum = Float(name="momentum", bounds=[0.1, 0.99])
        alpha = Float(name="alpha", bounds=[1e-6, 1e-1], log=True)
        cs.add([num_layers, max_units, learning_rate, batch_size, momentum, alpha])
        return cs

    def get_internal_optuna_search_space(self, **kwargs):
        """
        Get the internal Optuna search space.
        """
        n_train_samples = kwargs["n_train_samples"]
        # maximum batch size is the largest power of 2 that is smaller than the number of training samples
        max_batch_size_log2 = int(np.log2(n_train_samples))
        internal_optuna_search_space = {
            "num_layers": IntDistribution(low=1, high=5, step=1),
            "max_units": CategoricalDistribution(choices=[2**i for i in range(6, 10)]),
            "learning_rate": FloatDistribution(
                low=1e-4, high=1e-1, step=None, log=True
            ),
            "batch_size": CategoricalDistribution(
                choices=[2**i for i in range(4, max_batch_size_log2 + 1)]
            ),
            "momentum": FloatDistribution(low=0.1, high=0.99, step=None),
            "alpha": FloatDistribution(low=1e-6, high=1e-1, step=None, log=True),
        }
        return internal_optuna_search_space

    def construct_classifier(self, trial: Trial, **kwargs) -> None:
        """
        Construct the classifier based on the trial.
        """
        # search space somewhat similar to lcbench but no early stopping
        # missing dropout but included alpha
        n_train_samples = kwargs["n_train_samples"]
        if self.default:
            num_layers = 3
            max_units = 128
            learning_rate = 0.001
            batch_size = 32
            momentum = 0.9
            alpha = 0.0001
        else:
            num_layers = trial.suggest_int("num_layers", low=1, high=5)
            max_units = trial.suggest_categorical(
                "max_units", [2**i for i in range(6, 10)]
            )
            learning_rate = trial.suggest_float(
                "learning_rate", low=1e-4, high=1e-1, log=True
            )
            # maximum batch size is the largest power of 2 that is smaller than the number of training samples
            max_batch_size_log2 = int(np.log2(n_train_samples))
            batch_size = trial.suggest_categorical(
                "batch_size", [2**i for i in range(4, max_batch_size_log2 + 1)]
            )
            momentum = trial.suggest_float("momentum", low=0.1, high=0.99)
            alpha = trial.suggest_float("alpha", low=1e-6, high=1e-1, log=True)

        # hidden_layer_size is determined by num_layers and max_units
        # we start with max_units and half the number of units for each layer to create a funnel
        hidden_layer_sizes = [int(max_units / (2**i)) for i in range(num_layers)]

        classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="sgd",
            alpha=alpha,
            batch_size=batch_size,
            learning_rate="constant",
            learning_rate_init=learning_rate,
            max_iter=100,
            momentum=momentum,
            nesterovs_momentum=True,
            n_iter_no_change=100,
            verbose=False,
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
