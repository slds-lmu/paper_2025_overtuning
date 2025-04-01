import random
import warnings
from typing import Optional, Sequence

import numpy as np
import optuna
import pandas as pd
import torch
from hebo.acquisitions.acq import Mean, Sigma
from hebo.design_space import DesignSpace
from hebo.models.model_factory import get_model
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

from overtunebench.samplers import HeboSampler


class HeboSamplerMakarova(HeboSampler):
    """
    Wrapping HEBO as an Optuna sampler.
    Checks whether early stopping according to Makarova et al. (2022) is triggered.
    """

    def __init__(
        self,
        seed: int,
        hebo_search_space: DesignSpace,
        internal_optuna_search_space: dict,
        cv_metric: str,
    ):
        super().__init__(
            seed=seed,
            hebo_search_space=hebo_search_space,
            internal_optuna_search_space=internal_optuna_search_space,
        )
        self.cv_metric = cv_metric
        self.early_stopping_triggered = False
        self.regret = []  # lag of 1 with respect to the trial number
        self.sd_cv_values_corrected = []  # lag of 1 with respect to the trial number

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self.fallback_triggered = False

        # Check whether early stopping is triggered
        self.early_stopping_triggered = False
        if self.current_trial >= 20:
            try:
                # Early stopping starts after 20 trials (counting from 0)
                # Note: output transformations can mess up things here, therefore we do not use them for now
                #        in the original paper they apparently use log trafos for the output but even if the crossvalidated estimates are also transformed
                #        the Nadeau & Bengio (2003) correction must no longer be sensible on the transformed scale
                X, Xe = self.optimizer.space.transform(self.optimizer.X)
                # try:
                #    if self.optimizer.y.min() <= 0:
                #        y = torch.FloatTensor(power_transform(self.optimizer.y / self.optimizer.y.std(), method='yeo-johnson'))
                #    else:
                #        y = torch.FloatTensor(power_transform(self.optimizer.y / self.optimizer.y.std(), method='box-cox'))
                #        if y.std() < 0.5:
                #            y = torch.FloatTensor(power_transform(self.optimizer.y / self.optimizer.y.std(), method='yeo-johnson'))
                #    if y.std() < 0.5:
                #        raise RuntimeError('Power transformation failed')
                #    model = get_model(self.optimizer.model_name, self.optimizer.space.num_numeric, self.optimizer.space.num_categorical, 1, **self.optimizer.model_config)
                #    model.fit(X, Xe, y)
                # except:
                #    y     = torch.FloatTensor(self.y).clone()
                #    model = get_model(self.optimizer.model_name, self.optimizer.space.num_numeric, self.optimizer.space.num_categorical, 1, **self.optimizer.model_config)
                #    model.fit(X, Xe, y)

                y = torch.FloatTensor(self.optimizer.y).clone()
                model = get_model(
                    self.optimizer.model_name,
                    self.optimizer.space.num_numeric,
                    self.optimizer.space.num_categorical,
                    1,
                    **self.optimizer.model_config,
                )
                model.fit(X, Xe, y)

                mu = Mean(model)
                sig = Sigma(model)  # internally calculated as -Sigma

                # https://github.com/amazon-science/bo-early-stopping/blob/cffd7d367b5a3fc2abd1ba045300bb5aae29459b/src/enhanced_gp.py#L122
                t = X.shape[0]
                delta = 0.1
                beta_t = (
                    2
                    * np.log(
                        self.optimizer.space.num_paras
                        * (t**2)
                        * (np.pi**2)
                        / (6 * delta)
                    )
                    * 0.2
                )
                sqrt_beta_t = np.sqrt(beta_t)
                mus_X = mu(X, Xe)
                sigs_X = -sig(X, Xe)  # internally calculated as -Sigma
                ucbs_X = mus_X + sqrt_beta_t * sigs_X
                min_ucbs_X = ucbs_X.min().item()

                # https://github.com/amazon-science/bo-early-stopping/blob/cffd7d367b5a3fc2abd1ba045300bb5aae29459b/src/enhanced_gp.py#L22
                space_X = self.optimizer.quasi_sample(
                    n=2000
                )  # 2000 is the default number of samples in the original code
                space_X, space_Xe = self.optimizer.space.transform(space_X)
                mus_space_X = mu(space_X, space_Xe)
                sigs_space_X = -sig(
                    space_X, space_Xe
                )  # internally calculated as -Sigma
                lcbs_space_X = mus_space_X - sqrt_beta_t * sigs_space_X
                min_lcbs_space_X = lcbs_space_X.min().item()

                regret = min_ucbs_X - min_lcbs_space_X
                incumbent_trial = study.best_trial
                cv_values = eval(incumbent_trial.user_attrs[f"{self.cv_metric}_valid"])
                sd_cv_values = np.sqrt(np.var(cv_values))
                n_folds = len(cv_values)
                decay = np.sqrt(
                    (1 / n_folds + 1 / (n_folds - 1))
                )  # Nadeau & Bengio (2003) correction
                sd_cv_values_corrected = sd_cv_values * decay

                self.regret.append(regret)
                self.sd_cv_values_corrected.append(sd_cv_values_corrected)

                if regret <= sd_cv_values_corrected:
                    self.early_stopping_triggered = True
            except Exception as e:
                warnings.warn(
                    f"Early stopping checking failed with error: {e}. Falling back to no early stopping."
                )
                self.regret.append(np.nan)
                self.sd_cv_values_corrected.append(np.nan)
        else:
            self.regret.append(np.nan)
            self.sd_cv_values_corrected.append(np.nan)
