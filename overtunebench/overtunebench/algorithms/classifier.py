import warnings
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
from optuna.trial import Trial
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from overtunebench.utils import check_y_predict_proba


class Classifier(ABC):
    """
    Abstract class for a classifier.
    """

    def __init__(
        self,
        classifier_id: str,
        impute_x_cat: bool,
        impute_x_num: bool,
        encode_x: bool,
        scale_x: bool,
        seed: int,
        default: bool = False,
    ):
        self.classifier_id = classifier_id
        self.impute_x_cat = impute_x_cat
        self.impute_x_num = impute_x_num
        self.encode_x = encode_x
        self.scale_x = scale_x
        self.seed = seed
        self.default = default
        self.error_on_fit = False
        self.labels = None
        self.labels_fraction = None
        self.majority_class = None
        self.imputer_cat = None
        self.imputer_num = None
        self.encoder = None
        self.scaler = None
        self.classifier = None
        self.preprocessor = None

    @abstractmethod
    def get_hebo_search_space(self, **kwargs):
        """
        Get the HEBO search space.
        """
        pass

    @abstractmethod
    def get_configspace_search_space(self, **kwargs):
        """
        Get the configspace search space.
        """
        pass

    @abstractmethod
    def get_internal_optuna_search_space(self, **kwargs):
        """
        Get the internal Optuna search space.
        """
        pass

    def construct_pipeline(
        self,
        trial: Trial,
        refit: bool,
        cat_features: List[int],
        num_features: List[int],
        **kwargs
    ) -> None:
        """
        Construct the classifier pipeline based on the trial.
        First constructs the classifier, then the preprocessor.
        The preprocessor consists of imputing, encoding and scaling.
        If impute_x_cat is True, then categorical features are imputed with a constant value "__missing__".
        If not, then categorical features are passed through.
        If impute_x_num is True, then numerical features are imputed with the mean.
        If not, then numerical features are passed through.
        If encode_x is True, then categorical features are target encoded.
        Conditional on encode_x being True, if scale_x is True, then numerical features are scaled with a standard scaler.
        This means that all features are scaled to have a mean of zero and a standard deviance of 1 because the categorical features are target encoded and count as numerical features.
        Conditional on encode_x being True, if scale_x is False, then numerical features are passed through.
        If encode_x is False, then categorical features are passed through.
        Conditional on encode_x being False, if scale_x is True, then numerical features are scaled with a standard scaler.
        In this case the categorical features are still categorical and are not target encoded and therefore also not scaled as they do not count as numerical features.
        Otherwise, if scale_x is False, then numerical features are passed through.
        """
        if refit:
            self.construct_classifier_refit(trial, **kwargs)
        else:
            self.construct_classifier(trial, **kwargs)

        if self.impute_x_cat:
            self.imputer_cat = ColumnTransformer(
                transformers=[
                    (
                        "imputer_cat",
                        SimpleImputer(strategy="constant", fill_value="__missing__"),
                        cat_features,
                    )
                ],
                remainder="passthrough",
            )
            # cat_features are now the first features in the preprocessed data
            cat_features = list(range(len(cat_features)))
            # num features are the last features due to the passthrough
            num_features = list(
                range(len(cat_features), len(cat_features) + len(num_features))
            )
        else:
            self.imputer_cat = ColumnTransformer(
                transformers=[("passthrough", "passthrough", slice(0, None))],
                remainder="passthrough",
            )
        if self.impute_x_num:
            self.imputer_num = ColumnTransformer(
                transformers=[
                    (
                        "imputer_num",
                        SimpleImputer(strategy="mean"),
                        num_features,
                    )
                ],
                remainder="passthrough",
            )
            # num features are now the first features in the preprocessed data
            num_features = list(range(len(num_features)))
            # cat_features are the last features due to the passthrough
            cat_features = list(
                range(len(num_features), len(num_features) + len(cat_features))
            )
        else:
            self.imputer_num = ColumnTransformer(
                transformers=[("passthrough", "passthrough", slice(0, None))],
                remainder="passthrough",
            )
        if self.encode_x:
            self.encoder = ColumnTransformer(
                transformers=[
                    (
                        "encoder",
                        TargetEncoder(cv=5, shuffle=True, random_state=self.seed),
                        cat_features,
                    )
                ],
                remainder="passthrough",
            )
            if self.scale_x:
                self.scaler = ColumnTransformer(
                    transformers=[
                        (
                            "scaler",
                            StandardScaler(),
                            slice(0, None),
                        )
                    ],
                    remainder="passthrough",
                )
            else:
                self.scaler = ColumnTransformer(
                    transformers=[("passthrough", "passthrough", slice(0, None))],
                    remainder="passthrough",
                )
        else:
            self.encoder = ColumnTransformer(
                transformers=[("passthrough", "passthrough", slice(0, None))],
                remainder="passthrough",
            )
            if self.scale_x:
                self.scaler = ColumnTransformer(
                    transformers=[
                        (
                            "scaler",
                            StandardScaler(),
                            num_features,
                        )
                    ],
                    remainder="passthrough",
                )
            else:
                self.scaler = ColumnTransformer(
                    transformers=[("passthrough", "passthrough", slice(0, None))],
                    remainder="passthrough",
                )

        self.preprocessor = Pipeline(
            [
                ("imputer_cat", self.imputer_cat),
                ("imputer_num", self.imputer_num),
                ("encoder", self.encoder),
                ("scaler", self.scaler),
            ]
        )

    @abstractmethod
    def construct_classifier(self, trial: Trial, **kwargs) -> None:
        """
        Construct the classifier based on the trial.
        """
        pass

    @abstractmethod
    def construct_classifier_refit(self, trial: Trial, **kwargs) -> None:
        """
        Construct the classifier for refitting.
        """
        pass

    def fit(
        self,
        trial: Trial,
        x_train: pd.DataFrame,
        y_train: np.array,
        x_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[np.array] = None,
        cat_features: Optional[List[int]] = None,
    ) -> None:
        """
        Train the classifier pipeline.
        Performs the preprocessing and then calls _fit to train the classifier.
        """
        x_train = self.preprocessor.fit_transform(x_train, y_train)
        if x_valid is not None:
            x_valid = self.preprocessor.transform(x_valid)
        if cat_features is not None and len(cat_features) > 0:
            # if encode_x is True, then cat_features are now encoded and numeric
            if self.encode_x:
                preprocessed_cat_features = None
            # if encode_x is False, then cat_features are still categorical, and we need to find the indices of the cat_features among the preprocessed features
            else:
                preprocessed_features = self.preprocessor.get_feature_names_out(
                    self.preprocessor.feature_names_in_
                )
                preprocessed_features = [
                    feature.split("__")[-1] for feature in preprocessed_features
                ]
                if set(self.preprocessor.feature_names_in_).difference(
                    preprocessed_features
                ):
                    raise ValueError(
                        "Preprocessed features do not match input features"
                    )
                preprocessed_cat_features = []
                for cat_feature in cat_features:
                    cat_feature_name = self.preprocessor.feature_names_in_[cat_feature]
                    preprocessed_cat_feature = [
                        index
                        for index, feature_name in enumerate(preprocessed_features)
                        if cat_feature_name == feature_name
                    ]
                    preprocessed_cat_features.extend(preprocessed_cat_feature)
                if len(preprocessed_cat_features) != len(cat_features):
                    raise ValueError(
                        "Preprocessed categorical features do not match input categorical features"
                    )
        else:
            preprocessed_cat_features = None

        try:
            self._fit(
                trial,
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                cat_features=preprocessed_cat_features,
            )
        except Exception as e:
            warnings.warn("Exception occurred while fitting classifier: " + str(e))
            self.error_on_fit = True
            self.labels = np.unique(y_train).tolist()
            self.labels_fraction = np.bincount(y_train) / len(y_train)
            self.majority_class = np.bincount(y_train).argmax()

    @abstractmethod
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
        """
        pass

    def predict(self, x: pd.DataFrame) -> np.array:
        """
        Predict the class labels.
        """
        if self.error_on_fit:
            warnings.warn(
                "Classifier was not fitted successfully. Predicting majority class."
            )
            return np.repeat(self.majority_class, len(x))
        else:
            x = self.preprocessor.transform(x)
            y_pred_proba = self.classifier.predict_proba(x)
            y_pred = np.argmax(y_pred_proba, axis=1)
            return y_pred

    def predict_proba(self, x: pd.DataFrame) -> np.array:
        """
        Predict the class probabilities.
        """
        if self.error_on_fit:
            warnings.warn(
                "Classifier was not fitted successfully. Predicting class probabilities according to occurrence in training data."
            )
            y_pred_proba = np.tile(self.labels_fraction, (len(x), 1))
        else:
            x = self.preprocessor.transform(x)
            y_pred_proba = self.classifier.predict_proba(x)
        check_y_predict_proba(y_pred_proba)
        return y_pred_proba

    def reset(self) -> None:
        """
        Reset the classifier and preprocessor and set error_on_fit to False, labels to None and majority_class to None.
        """
        self.error_on_fit = False
        self.labels = None
        self.labels_fraction = None
        self.majority_class = None
        self.preprocessor = None
        self.classifier = None
