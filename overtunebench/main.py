if __name__ == "__main__":
    import argparse

    from overtunebench.algorithms import (
        CatBoost,
        Featureless,
        FunnelMLP,
        LogReg,
        TabPFN,
        XGBoost,
        XGBoostLarge,
    )
    from overtunebench.learner import (
        LearnerHeboCV,
        LearnerHeboHoldout,
        LearnerHeboMakarovaCV,
        LearnerHeboRepeatedHoldout,
        LearnerRandomCV,
        LearnerRandomHoldout,
        LearnerRandomRepeatedHoldout,
        LearnerSmacCV,
        LearnerSmacHoldout,
        LearnerSmacRepeatedHoldout,
    )
    from overtunebench.utils import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier",
        type=str,
        default="catboost",
        choices=[
            "catboost",
            "funnel_mlp",
            "logreg",
            "tabpfn",
            "xgboost",
            "xgboost_large",
            "featureless",
        ],
    )
    parser.add_argument("--default", type=str2bool, default=False)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="random",
        choices=["random", "hebo", "hebo_makarova", "smac"],
    )
    parser.add_argument(
        "--data_id",
        type=int,
        default=11111,
        choices=[
            23517,
            1169,
            41147,
            4135,
            1461,
            1590,
            41150,
            41162,
            42733,
            42742,
            99999,
            11111,
        ],
    )
    parser.add_argument(
        "--valid_type",
        type=str,
        default="holdout",
        choices=["cv", "holdout", "repeatedholdout"],
    )
    parser.add_argument(
        "--train_valid_size",
        type=int,
        default=500,
        choices=[500, 1000, 5000],
    )
    parser.add_argument("--reshuffle", type=str2bool, default=True)
    parser.add_argument("--n_splits", type=int, default=5, choices=[5])
    # n_repeats = 1 or 5 for cv, 1 for holdout and 5 for repeatedholdout
    parser.add_argument("--n_repeats", type=int, default=1, choices=[1, 5])
    parser.add_argument("--valid_frac", type=float, default=0.2, choices=[0.2])
    parser.add_argument("--test_size", type=int, default=5000)
    parser.add_argument("--add_valid_size", type=int, default=5000)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    classifiers = {
        "catboost": CatBoost(seed=args.seed, default=args.default),
        "funnel_mlp": FunnelMLP(seed=args.seed, default=args.default),
        "logreg": LogReg(seed=args.seed, default=args.default),
        "tabpfn": TabPFN(seed=args.seed),
        "xgboost": XGBoost(seed=args.seed, default=args.default),
        "xgboost_large": XGBoostLarge(seed=args.seed, default=args.default),
        "featureless": Featureless(seed=args.seed),
    }
    classifier = classifiers[args.classifier]

    if args.n_trials > 500:
        raise ValueError(
            "n_trials must be <= 500 - or you must adjust seeds in codebase"
        )

    if args.optimizer == "random":
        if args.valid_type == "cv":
            learner = LearnerRandomCV(
                classifier=classifier,
                data_id=args.data_id,
                train_valid_size=args.train_valid_size,
                reshuffle=args.reshuffle,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                test_size=args.test_size,
                add_valid_size=args.add_valid_size,
                n_trials=args.n_trials,
                seed=args.seed,
            )
        elif args.valid_type == "holdout":
            learner = LearnerRandomHoldout(
                classifier=classifier,
                data_id=args.data_id,
                train_valid_size=args.train_valid_size,
                reshuffle=args.reshuffle,
                valid_frac=args.valid_frac,
                test_size=args.test_size,
                add_valid_size=args.add_valid_size,
                n_trials=args.n_trials,
                seed=args.seed,
            )
        else:
            learner = LearnerRandomRepeatedHoldout(
                classifier=classifier,
                data_id=args.data_id,
                train_valid_size=args.train_valid_size,
                reshuffle=args.reshuffle,
                valid_frac=args.valid_frac,
                n_repeats=args.n_repeats,
                test_size=args.test_size,
                add_valid_size=args.add_valid_size,
                n_trials=args.n_trials,
                seed=args.seed,
            )
    elif args.optimizer == "hebo_makarova":
        if args.valid_type == "cv" and args.n_repeats == 1:
            learner = LearnerHeboMakarovaCV(
                classifier=classifier,
                metric="auc",
                data_id=args.data_id,
                train_valid_size=args.train_valid_size,
                reshuffle=args.reshuffle,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                test_size=args.test_size,
                add_valid_size=args.add_valid_size,
                n_trials=args.n_trials,
                seed=args.seed,
            )
        else:
            raise ValueError(
                "Hebo with early stopping a la Makarova et al. (2022) is only implemented for standard CV"
            )
    elif args.optimizer == "hebo":
        if args.valid_type == "cv":
            learner = LearnerHeboCV(
                classifier=classifier,
                metric="auc",
                data_id=args.data_id,
                train_valid_size=args.train_valid_size,
                reshuffle=args.reshuffle,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                test_size=args.test_size,
                add_valid_size=args.add_valid_size,
                n_trials=args.n_trials,
                seed=args.seed,
            )
        elif args.valid_type == "holdout":
            learner = LearnerHeboHoldout(
                classifier=classifier,
                metric="auc",
                data_id=args.data_id,
                train_valid_size=args.train_valid_size,
                reshuffle=args.reshuffle,
                valid_frac=args.valid_frac,
                test_size=args.test_size,
                add_valid_size=args.add_valid_size,
                n_trials=args.n_trials,
                seed=args.seed,
            )
        else:
            learner = LearnerHeboRepeatedHoldout(
                classifier=classifier,
                metric="auc",
                data_id=args.data_id,
                train_valid_size=args.train_valid_size,
                reshuffle=args.reshuffle,
                valid_frac=args.valid_frac,
                n_repeats=args.n_repeats,
                test_size=args.test_size,
                add_valid_size=args.add_valid_size,
                n_trials=args.n_trials,
                seed=args.seed,
            )
    elif args.optimizer == "smac":
        if args.valid_type == "cv":
            learner = LearnerSmacCV(
                classifier=classifier,
                metric="auc",
                data_id=args.data_id,
                train_valid_size=args.train_valid_size,
                reshuffle=args.reshuffle,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                test_size=args.test_size,
                add_valid_size=args.add_valid_size,
                n_trials=args.n_trials,
                seed=args.seed,
            )
        elif args.valid_type == "holdout":
            learner = LearnerSmacHoldout(
                classifier=classifier,
                metric="auc",
                data_id=args.data_id,
                train_valid_size=args.train_valid_size,
                reshuffle=args.reshuffle,
                valid_frac=args.valid_frac,
                test_size=args.test_size,
                add_valid_size=args.add_valid_size,
                n_trials=args.n_trials,
                seed=args.seed,
            )
        else:
            learner = LearnerSmacRepeatedHoldout(
                classifier=classifier,
                metric="auc",
                data_id=args.data_id,
                train_valid_size=args.train_valid_size,
                reshuffle=args.reshuffle,
                valid_frac=args.valid_frac,
                n_repeats=args.n_repeats,
                test_size=args.test_size,
                add_valid_size=args.add_valid_size,
                n_trials=args.n_trials,
                seed=args.seed,
            )
    else:
        raise ValueError("Invalid optimizer")
    learner.run()
