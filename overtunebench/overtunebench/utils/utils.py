import argparse
import warnings
from json import JSONEncoder
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.utils import resample

from overtunebench.metrics import compute_metric


class NumpyArrayEncoder(JSONEncoder):
    """
    Encode numpy arrays to lists.
    """

    def default(self, obj: object) -> object:
        """
        Encode numpy arrays to lists.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def str2bool(value: str) -> bool:
    """
    Convert a string to a boolean.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def int_or_none(value: str) -> int or None:
    """
    Convert a string to an integer or None.
    """
    try:
        return int(value)
    except ValueError:
        if value.lower() == "none":
            return None
        else:
            raise argparse.ArgumentTypeError(f"{value} is not a valid integer or None")


def save_single_array(file_path: str, data: np.array) -> None:
    """
    Save a single 1-dimensional numpy array to a parquet file.
    This is usually an integer array of the target variable.
    """
    pa_table = pa.Table.from_arrays(
        [pa.array(data.tolist(), type=pa.int32())], names=["data"]
    )
    pq.write_table(pa_table, file_path)


def load_single_array(file_path: str) -> List:
    """
    Load a single 1-dimensional numpy array from a parquet file.
    This is usually an integer array of the target variable.
    """
    table = pq.read_table(file_path, use_threads=False, memory_map=True)
    reconstructed_data = np.array(table.column(0).to_pylist(), dtype=np.int32)
    return reconstructed_data


def save_list_of_1d_arrays(file_path: str, data: List) -> None:
    """
    Save a list of 1-dimensional numpy arrays to a parquet file.
    These are usually integer arrays of the target variable.
    """
    flattened_data = []
    column_names = []
    for i, numpy_array in enumerate(data):
        flattened_data.append(numpy_array.tolist())
        column_names.append(f"data_{i}")
    arrays = [pa.array(column, type=pa.int32()) for column in flattened_data]
    pa_table = pa.Table.from_arrays(arrays, names=column_names)
    pq.write_table(pa_table, file_path)


def load_list_of_1d_arrays(file_path: str) -> List:
    """
    Load a list of 1-dimensional numpy arrays from a parquet file.
    These are usually integer arrays of the target variable.
    """
    df = pd.read_parquet(
        file_path, engine="pyarrow", use_threads=False, memory_map=True
    )
    reconstructed_data = []
    for column in df.columns:
        numpy_array = np.array(df[column].tolist(), dtype=np.int32)
        reconstructed_data.append(numpy_array)
    return reconstructed_data


def save_list_of_pd_arrays(file_path: str, data: List) -> None:
    """
    Save a list of p-dimensional (n times p) numpy arrays to a parquet file.
    These are usually probability arrays.
    """
    flattened_data = []
    column_names = []
    types = []
    for i, numpy_array in enumerate(data):
        for column in range(numpy_array.shape[1]):
            types.append(
                pa.float64() if numpy_array.dtype == np.float64 else pa.float32()
            )
            flattened_data.append(numpy_array[:, column].tolist())
            column_names.append(f"data_{i}_column_{column}")
    if not all(types[0] == t for t in types):
        raise ValueError("All types must be the same")
    arrays = [pa.array(column, type=types[0]) for column in flattened_data]
    pa_table = pa.Table.from_arrays(arrays, names=column_names)
    pq.write_table(pa_table, file_path)


def load_list_of_pd_arrays(file_path: str) -> List:
    """
    Load a list of p-dimensional (n times p) numpy arrays from a parquet file.
    These are usually probability arrays.
    """
    df = pd.read_parquet(
        file_path, engine="pyarrow", use_threads=False, memory_map=True
    )
    indices = set()
    for column in df.columns:
        i, _, _ = column.split("_")[1:4]
        indices.add((int(i)))
    reconstructed_data = []
    for i in sorted(indices):
        cols = [col for col in df.columns if col.startswith(f"data_{i}_")]
        cols.sort(
            key=lambda x: int(x.rsplit("_", 1)[-1])
        )  # robust sorting in the case of p > 10
        array = df[cols].to_numpy()
        reconstructed_data.append(array)
    return reconstructed_data


def save_list_of_list_of_1d_arrays(file_path: str, data: List) -> None:
    """
    Save a list of list of 1-dimensional numpy arrays to a parquet file.
    These are usually integer arrays of the target variable.
    """
    flattened_data = []
    column_names = []
    for i, sublist in enumerate(data):
        for j, numpy_array in enumerate(sublist):
            flattened_data.append(numpy_array.tolist())
            column_names.append(f"data_{i}_{j}")

    pa_table = pa.Table.from_arrays(
        [pa.array(column, type=pa.int32()) for column in flattened_data],
        names=column_names,
    )
    pq.write_table(pa_table, file_path)


def load_list_of_list_of_1d_arrays(file_path: str) -> List:
    """
    Load a list of list of 1-dimensional numpy arrays from a parquet file.
    These are usually integer arrays of the target variable.
    """
    df = pd.read_parquet(
        file_path, engine="pyarrow", use_threads=False, memory_map=True
    )
    max_i = max(int(col.split("_")[1]) for col in df.columns)
    reconstructed_data = [[] for _ in range(max_i + 1)]
    for column in df.columns:
        i, j = map(int, column.split("_")[1:3])
        numpy_array = np.array(df[column].tolist(), dtype=np.int32)
        reconstructed_data[i].append(numpy_array)
    return reconstructed_data


def save_list_of_list_of_pd_arrays(file_path: str, data: List) -> None:
    """
    Save a list of list of p-dimensional (n times p) numpy arrays to a parquet file.
    These are usually probability arrays.
    """
    flattened_data = []
    column_names = []
    types = []
    for i, sublist in enumerate(data):
        for j, numpy_array in enumerate(sublist):
            for column in range(numpy_array.shape[1]):
                types.append(
                    pa.float64() if numpy_array.dtype == np.float64 else pa.float32()
                )
                flattened_data.append(numpy_array[:, column].tolist())
                column_names.append(f"data_{i}_{j}_column_{column}")
    if not all(types[0] == t for t in types):
        raise ValueError("All types must be the same")
    arrays = [pa.array(column, type=types[0]) for column in flattened_data]
    pa_table = pa.Table.from_arrays(arrays, names=column_names)
    pq.write_table(pa_table, file_path)


def load_list_of_list_of_pd_arrays(file_path: str) -> List:
    """
    Load a list of list of p-dimensional (n times p) numpy arrays from a parquet file.
    These are usually probability arrays.
    """
    df = pd.read_parquet(
        file_path, engine="pyarrow", use_threads=False, memory_map=True
    )
    indices = set()
    for column in df.columns:
        i, j, _ = column.split("_")[1:4]
        indices.add((int(i), int(j)))
    reconstructed_data = []
    for i, j in sorted(indices):
        while len(reconstructed_data) <= i:
            reconstructed_data.append([])
        cols = [col for col in df.columns if col.startswith(f"data_{i}_{j}_")]
        cols.sort(
            key=lambda x: int(x.rsplit("_", 1)[-1])
        )  # robust sorting in the case of p > 10
        array = df[cols].to_numpy()
        reconstructed_data[i].append(array)
    return reconstructed_data


def unify_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unify missing values in a DataFrame based on the dtype of each column.
    This does not impute missing values but rather brings them into a standardized format.
    """
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].fillna("NaN")
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(np.nan)
        else:
            raise ValueError(f"Unsupported dtype: {df[col].dtype}")
    return df


def construct_x_and_y_add_valid(
    x_valid: pd.DataFrame,
    y_valid: np.array,
    x_add_valid: pd.DataFrame,
    y_add_valid: np.array,
) -> (pd.DataFrame, np.array):
    """
    Construct the additional validation data by combining the validation data and the additional validation data.
    """
    x_add_valid_final = pd.concat([x_valid, x_add_valid])
    y_add_valid_final = np.concatenate((y_valid, y_add_valid))
    return x_add_valid_final, y_add_valid_final


def check_y_predict_proba(y_predict_proba: np.array) -> np.array:
    """
    Check if a numpy array of probability predictions sums to 1.0.
    """
    row_sums = y_predict_proba.sum(1)
    if any(row_sums != 1.0):
        largest_absolute_difference = np.max(np.abs(row_sums - 1.0))
        warnings.warn(
            f"Probabilities do not sum to 1.0. Largest absolute difference: {largest_absolute_difference}"
        )
        if largest_absolute_difference > 1e-6:
            raise ValueError("Probabilities do not sum to 1.0")
    return y_predict_proba


def bootstrap_test_performance(
    y_test: np.array,
    y_pred: np.array,
    y_pred_proba: np.array,
    metric: str,
    labels: list,
    multiclass: bool,
    seed: int,
    n_samples: int = 10,
) -> List:
    """
    Bootstrap the test performance of a model.
    """
    metric_test_bootstrap = []
    n_test = len(y_test)
    for b in range(n_samples):
        idx = resample(
            list(range(n_test)),
            replace=True,
            n_samples=n_test,
            random_state=seed + b,
            stratify=y_test,
        )
        metric_test_bootstrap.append(
            compute_metric(
                y_test[idx],
                y_pred=y_pred[idx],
                y_pred_proba=y_pred_proba[idx, :],
                metric=metric,
                labels=labels,
                multiclass=multiclass,
            )
        )
    return metric_test_bootstrap
