import random
from typing import Optional, Sequence

import numpy as np
import optuna
import pandas as pd
import torch
from ConfigSpace.configuration_space import ConfigurationSpace
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialValue


# https://automl.github.io/SMAC3/main/examples/1_basics/3_ask_and_tell.html#ask-and-tell
class SmacSampler(optuna.samplers.BaseSampler):
    """
    Wrapping SMAC as an Optuna sampler.
    """

    def __init__(
        self,
        seed: int,
        configspace_search_space: ConfigurationSpace,
        internal_optuna_search_space: dict,
        n_trials: int,
        fid: str,
    ):
        self.seed = seed
        self.configspace_search_space = configspace_search_space
        self.internal_optuna_search_space = internal_optuna_search_space
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.scenario = Scenario(
            self.configspace_search_space,
            deterministic=False,
            n_trials=n_trials,
            seed=self.seed,
        )
        self.intensifier = HyperparameterOptimizationFacade.get_intensifier(
            self.scenario, max_config_calls=1
        )
        self.smac = HyperparameterOptimizationFacade(
            self.scenario,
            target_function=fid,
            intensifier=self.intensifier,
            overwrite=True,
        )
        self.current_trial = 0
        self.info_history = []
        self.params_history = []
        self.values_history = []
        self.fallback_triggered = False

    def sample_relative(self, study, trial, search_space):
        info = self.smac.ask()
        params = pd.DataFrame(info.config.get_dictionary(), index=[0])
        params.reset_index(drop=True, inplace=True)
        self.info_history.append(info)
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
        # Note: SMAC assumes minimization therefore we correct for that here
        if state == TrialState.COMPLETE:
            if study.direction == optuna.study.StudyDirection.MAXIMIZE:
                self.values_history.append(-np.array(values))
            else:
                self.values_history.append(np.array(values))
        else:
            self.values_history.append(np.nan)  # FIXME: check
        value = TrialValue(cost=self.values_history[-1])
        self.smac.tell(self.info_history[-1], value=value)
