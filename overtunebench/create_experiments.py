if __name__ == "__main__":
    import argparse
    import os

    from overtunebench.utils import str2bool

    data_ids = [23517, 1169, 41147, 4135, 1461, 1590, 41150, 41162, 42733, 42742]
    train_valid_sizes = [500, 1000, 5000]
    reshuffle = [True, False]
    seeds = range(42, 52)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier",
        type=str,
        choices=[
            "catboost",
            "funnel_mlp",
            "logreg",
            "tabpfn",
            "xgboost",
            "xgboost_large",
        ],
        required=True,
    )
    parser.add_argument("--default", type=str2bool, required=True)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["random", "hebo", "hebo_makarova", "smac"],
        required=True,
    )
    parser.add_argument(
        "--valid_type",
        type=str,
        choices=["cv", "holdout", "repeatedholdout"],
        required=True,
    )
    parser.add_argument("--n_repeats", type=int, default=None)
    args = parser.parse_args()

    if args.optimizer in ["hebo", "hebo_makarova", "smac"]:
        n_trials = 250
        if args.valid_type == "cv":
            n_repeats = args.n_repeats
        elif args.valid_type == "repeatedholdout":
            n_repeats = 5
    else:
        n_trials = 500
        if args.valid_type == "cv":
            n_repeats = 5
        elif args.valid_type == "repeatedholdout":
            n_repeats = 5

    file = "run_experiments.sh"
    with open(file, "+w") as f:
        for data_id in data_ids:
            for train_valid_size in train_valid_sizes:
                for reshuffle_ in reshuffle:
                    for seed in seeds:
                        if args.valid_type == "cv":
                            f.write(
                                f"python3 main.py --classifier {args.classifier} --default {args.default} --optimizer {args.optimizer} --data_id {data_id} --valid_type {args.valid_type} --train_valid_size {train_valid_size} --reshuffle {reshuffle_} --n_splits 5 --n_repeats {n_repeats} --test_size 5000 --add_valid_size 5000 --seed {seed} --n_trials {n_trials} \n"
                            )
                        elif args.valid_type == "holdout":
                            f.write(
                                f"python3 main.py --classifier {args.classifier} --default {args.default} --optimizer {args.optimizer} --data_id {data_id} --valid_type {args.valid_type} --train_valid_size {train_valid_size} --reshuffle {reshuffle_} --valid_frac 0.2 --test_size 5000 --add_valid_size 5000 --seed {seed} --n_trials {n_trials} \n"
                            )
                        else:
                            f.write(
                                f"python3 main.py --classifier {args.classifier} --default {args.default} --optimizer {args.optimizer} --data_id {data_id} --valid_type {args.valid_type} --train_valid_size {train_valid_size} --reshuffle {reshuffle_} --valid_frac 0.2 --n_repeats {n_repeats} --test_size 5000 --add_valid_size 5000 --seed {seed} --n_trials {n_trials} \n"
                            )

    os.chmod(file, 0o755)
