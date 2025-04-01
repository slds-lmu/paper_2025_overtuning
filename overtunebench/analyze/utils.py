from typing import List


def compute_overtuning(test_values: List[float]) -> List[float]:
    """
    Compute the overtuning metric.
    """
    best_test_value = float("inf")
    overtuning = []
    for i in range(len(test_values)):
        test_value = test_values[i]
        if test_value < best_test_value:
            best_test_value = test_value
        overtuning.append(-(best_test_value - test_value))
    return overtuning
