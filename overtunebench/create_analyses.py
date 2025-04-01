if __name__ == "__main__":
    import argparse
    import os

    from overtunebench.utils import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["random", "hebo", "hebo_makarova", "smac"],
        required=True,
    )
    parser.add_argument(
        "--valid_type",
        type=str,
        choices=["cv", "cv_repeated", "holdout", "repeatedholdout"],
        required=True,
    )
    parser.add_argument("--n_repeats", type=int, default=None)
    parser.add_argument(
        "--type",
        type=str,
        default="basic",
        choices=[
            "basic",
            "post_naive",
            "basic_simulate_repeatedholdout",
            "post_naive_simulate_repeatedholdout",
        ],
    )
    parser.add_argument(
        "--reshuffle", type=str, default="Both", choices=["True", "False", "Both"]
    )
    parser.add_argument("--check_files", type=str2bool, default=False)
    parser.add_argument("--correct_path", type=str2bool, default=False)
    args = parser.parse_args()

    results_subfolders = os.listdir("results")
    if args.optimizer in ["hebo", "hebo_makarova", "smac"]:
        valid_abbrev = {
            "cv": "_cv5r1_",
            "cv_repeated": "_cv5r5_",
            "holdout": "_holdout02_",
            "repeatedholdout": "_repeatedholdout02_",
        }
        valid = valid_abbrev[args.valid_type]
        results_subfolders = [
            results_subfolder
            for results_subfolder in results_subfolders
            if args.optimizer in results_subfolder and valid in results_subfolder
        ]
        if args.optimizer == "hebo":
            results_subfolders = [
                results_subfolder
                for results_subfolder in results_subfolders
                if "hebo_makarova" not in results_subfolder
                and valid in results_subfolder
            ]
    else:
        valid_abbrev = {
            "cv": "_cv5r5_",
            "cv_repeated": "_cv5r5_",
            "holdout": "_holdout02_",
            "repeatedholdout": "_cv5r5_",  # repeatedholdout simulated from repeated cv
        }
        valid = valid_abbrev[args.valid_type]
        results_subfolders = [
            results_subfolder
            for results_subfolder in results_subfolders
            if "hebo" not in results_subfolder
            and "smac" not in results_subfolder
            and valid in results_subfolder
        ]

    file = "run_analyses.sh"
    with open(file, "+w") as f:
        for results_subfolder in results_subfolders:
            if args.reshuffle == "True" or args.reshuffle == "Both":
                if "reshuffle" in results_subfolder:
                    f.write(
                        f"python3 result_analyzer.py --max_workers {args.max_workers} --type {args.type} --results_subfolder {results_subfolder} --n_repeats {args.n_repeats} --check_files {args.check_files} --correct_path {args.correct_path} \n"
                    )
            if args.reshuffle == "False" or args.reshuffle == "Both":
                if "reshuffle" not in results_subfolder:
                    f.write(
                        f"python3 result_analyzer.py --max_workers {args.max_workers} --type {args.type} --results_subfolder {results_subfolder} --n_repeats {args.n_repeats} --check_files {args.check_files} --correct_path {args.correct_path} \n"
                    )

    os.chmod(file, 0o755)
