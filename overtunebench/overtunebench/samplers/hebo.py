import random
import warnings
from typing import Optional, Sequence

import numpy as np
import optuna
import pandas as pd
import torch
from hebo.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState


class HeboSampler(optuna.samplers.BaseSampler):
    """
    Wrapping HEBO as an Optuna sampler.
    """

    def __init__(
        self,
        seed: int,
        hebo_search_space: DesignSpace,
        internal_optuna_search_space: dict,
    ):
        self.seed = seed
        self.hebo_search_space = hebo_search_space
        self.internal_optuna_search_space = internal_optuna_search_space
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)  # GPs
        self.optimizer = HEBO(hebo_search_space, scramble_seed=self.seed)
        self.current_trial = 0
        self.params_history = []
        self.values_history = []
        self.fallback_triggered = False

    def sample_relative(self, study, trial, search_space):
        # Note: HEBO apparently does not have its own fallback mechanism
        try:
            params = self.optimizer.suggest(1)
        except Exception as e:
            warnings.warn(
                f"HEBO failed with error: {e}. Falling back to random sampling."
            )
            self.fallback_triggered = True
            params = {}
            for param_name, param_distribution in search_space.items():
                params[param_name] = self.sample_param_random_fallback(
                    study=study,
                    trial=trial,
                    param_name=param_name,
                    param_distribution=param_distribution,
                )
            params = pd.DataFrame([params])
        params.reset_index(drop=True, inplace=True)
        self.params_history.append(params)
        self.current_trial += 1
        return {
            column: params.at[0, column] for column in params.columns
        }  # workaround to prevent pandas to drop dtype information

    def infer_relative_search_space(self, study, trial):
        # return optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))
        return self.internal_optuna_search_space

    def sample_independent(self, study, trial, param_name, param_distribution):
        self.fallback_triggered = True
        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def sample_param_random_fallback(
        self, study, trial, param_name, param_distribution
    ):
        independent_sampler = optuna.samplers.RandomSampler(seed=self.seed)
        return independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self.fallback_triggered = False

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        assert state in [TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED]
        # Note: This assumes single-objective optimization
        # Note: HEBO assumes minimization therefore we correct for that here
        if state == TrialState.COMPLETE:
            if study.direction == optuna.study.StudyDirection.MAXIMIZE:
                self.values_history.append(-np.array(values))
            else:
                self.values_history.append(np.array(values))
        else:
            self.values_history.append(np.nan)  # FIXME: check
        self.optimizer.observe(
            self.params_history[-1],
            self.values_history[-1],
        )
