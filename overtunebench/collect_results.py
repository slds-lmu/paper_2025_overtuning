if __name__ == "__main__":
    import argparse
    import os

    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--valid_type",
        type=str,
        choices=["cv", "cv_repeated", "holdout", "repeatedholdout"],
        required=True,
    )

    args = parser.parse_args()

    valid_abbrev = {
        "cv": ["_cv5r1ur1_", "_cv5r5ur1_"],
        "cv_repeated": ["_cv5r5ur5_"],
        "holdout": ["_holdout02_"],
        "repeatedholdout": ["_repeatedholdout02_5ur5_", "_simulate_repeatedholdout_"],
    }
    valid = valid_abbrev[args.valid_type]

    files = os.listdir("csvs/raw")
    files = [
        file
        for file in files
        if file.endswith(".csv") and any([v in file for v in valid])
    ]
    if args.valid_type == "cv" or args.valid_type == "cv_repeated":
        files = [file for file in files if "simulate_repeatedholdout" not in file]

    results_raw = [file for file in files if "results" in file and "raw" in file]
    curvature = [file for file in files if "curvature" in file]
    kendalls_tau_valid_test = [file for file in files if "kendalls_tau" in file]
    results_post = [file for file in files if "results" in file and "post" in file]

    if results_raw:
        results_raw_csv = pd.concat(
            [pd.read_csv(f"csvs/raw/{file}") for file in results_raw]
        )
        results_raw_csv.to_csv(f"csvs/results_{args.valid_type}.csv", index=False)

    if curvature:
        curvature_csv = pd.concat(
            [pd.read_csv(f"csvs/raw/{file}") for file in curvature]
        )
        curvature_csv.to_csv(f"csvs/curvature_{args.valid_type}.csv", index=False)

    if kendalls_tau_valid_test:
        for type in ["test.csv", "test_retrained.csv"]:
            kendalls_tau_valid_test_csv = pd.concat(
                [
                    pd.read_csv(f"csvs/raw/{file}")
                    for file in kendalls_tau_valid_test
                    if type in file
                ]
            )
            kendalls_tau_valid_test_csv.to_csv(
                f"csvs/kendalls_tau_{args.valid_type}_valid_test_{type}", index=False
            )

    if results_post:
        for type in ["test.csv", "test_retrained.csv"]:
            results_post_csv = pd.concat(
                [
                    pd.read_csv(f"csvs/raw/{file}")
                    for file in results_post
                    if type in file
                ]
            )
            results_post_csv.to_csv(
                f"csvs/results_{args.valid_type}_post_{type}", index=False
            )
