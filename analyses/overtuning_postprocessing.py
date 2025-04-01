import pandas as pd
import numpy as np


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# Note: robustify bootstrapping mechanisms
def compute_overtuning(subset: pd.DataFrame, valid_id: str, test_id: str, minimize: bool, epsilon: float=0.001, sample: bool=True, random_state: int=42, N: int=0, n_repeats: int=1) -> pd.DataFrame:
    if not minimize:
        subset[valid_id] = -subset[valid_id]
        subset[test_id] = -subset[test_id]
    # remove all rows with infinite or missing valid or test values
    subset = subset[~subset[[valid_id, test_id]].isnull().any(axis=1)]
    subset = subset[~subset[[valid_id, test_id]].isin([np.inf, -np.inf]).any(axis=1)]
    n = len(subset)
    if N == 0:
        N = n
    if n < N:
        raise ValueError(f"Number of rows {n} is less than the number of samples {N}")
    repeat_results = []
    for repeat in range(n_repeats):
        # permute the order of the rows
        if sample:
            subset_sample = subset.sample(frac=1, random_state=random_state).reset_index(drop=True)
        else:
            subset_sample = subset.reset_index(drop=True)
        # select the first N rows
        subset_sample = subset_sample.iloc[:N]
        default_test = subset_sample.iloc[0][test_id]
        # compute incumbent indices cumulatively
        valid_cummins = subset_sample[valid_id].expanding().apply(lambda x: x.idxmin(), raw=False)
        test_cummins = subset_sample[test_id].expanding().apply(lambda x: x.idxmin(), raw=False)
        incumbent = valid_cummins.astype(int).values
        incumbent_test = test_cummins.astype(int).values
        # compute valid and test performance of incumbents
        valid_incumbents = subset_sample.loc[incumbent, valid_id].values
        test_incumbents = subset_sample.loc[incumbent, test_id].values
        # compute best test performance over incumbents
        best_test_over_incumbents = subset_sample.loc[incumbent, test_id].expanding().min().values
        # calculate overtuning
        overtuning = subset_sample.loc[incumbent, test_id].values - best_test_over_incumbents
        overtuning = np.array(overtuning)
        denominator = default_test - best_test_over_incumbents
        denominator = np.array(denominator)
        overtuning_relative = overtuning / denominator
        # for the first iteration t = 1, the incumbent is the default configuration and overtuning is 0
        overtuning_relative[0] = 0
        # no progress means that the test performance of the default was the best so far
        # we track these pathological cases to exclude them from the analysis
        no_progress = (denominator < epsilon)
        overtuning_relative[no_progress] = np.nan
        # compute test regret
        best_test_over_incumbents_test = subset_sample.loc[incumbent_test, test_id].expanding().min().values
        test_regret = subset_sample.loc[incumbent, test_id].values - best_test_over_incumbents_test
        # compute meta overfitting
        meta_overfitting = subset_sample.loc[incumbent, test_id].values - subset_sample.loc[incumbent, valid_id].values
        final_iteration = np.repeat(False, N)
        final_iteration[-1] = True
        results = pd.DataFrame({"repeat": repeat, "iteration": range(1, N + 1), "valid_incumbent": valid_incumbents, "test_incumbent": test_incumbents, "best_test_over_incumbents": best_test_over_incumbents, "best_test_over_incumbents_test": best_test_over_incumbents_test, "overtuning": overtuning, "overtuning_relative": overtuning_relative, "no_progress": no_progress, "test_regret": test_regret, "meta_overfitting": meta_overfitting, "final_iteration": final_iteration})
        repeat_results.append(results)
    results = pd.concat(repeat_results)
    return results


if __name__ == "__main__":
    import pandas as pd
    import argparse
    np.seterr(divide="ignore", invalid="ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="pd1")
    parser.add_argument("--reshuffling_train_valid_size", type=int, default=500)
    parser.add_argument("--reshuffling_resampling", type=str, default="cv")
    parser.add_argument("--reshuffling_reshuffled", type=str2bool, default=False)
    parser.add_argument("--reshuffling_classifier", type=str, default="catboost")
    parser.add_argument("--reshuffling_optimizer", type=str, default="hebo_makarova")
    parser.add_argument("--reshuffling_retrained", type=str2bool, default=True)
    parser.add_argument("--wdtb_model", type=str, default="Resnet")
    parser.add_argument("--wdtb_subbenchmark", type=str, default="categorical_classification_medium")
    parser.add_argument("--tabzilla_model", type=str, default="CatBoost")
    parser.add_argument("--tabzilla_subbenchmark", type=str, default="binary")
    parser.add_argument("--tabrepo_model", type=str, default="CatBoost")
    parser.add_argument("--tabrepo_subbenchmark", type=str, default="binary")
    parser.add_argument("--sample", type=str2bool, default=False)
    parser.add_argument("--N", type=int, default=0)
    parser.add_argument("--n_repeats", type=int, default=1)
    args = parser.parse_args()


    if args.benchmark == "lcbench":
        results = []
        df = pd.read_csv("lcbench/data.csv")
        df = df[df["epoch"] == 50]
        scenario_ids = ["OpenML_task_id"]
        metric_dict = {
            "accuracy": {"valid_id": "val_accuracy", "test_id": "test_accuracy"},
            "balanced_accuracy": {"valid_id": "val_balanced_accuracy", "test_id": "test_balanced_accuracy"},
            "cross_entropy": {"valid_id": "val_cross_entropy", "test_id": "test_cross_entropy"}
        }
        minimize_dict = {
            "accuracy": False,
            "balanced_accuracy": False,
            "cross_entropy": True
        }
        rel_pathology_dict = {}
        for metric in ["accuracy", "balanced_accuracy", "cross_entropy"]:
            df_overtuning = df.groupby(scenario_ids).apply(
                compute_overtuning,
                valid_id=metric_dict[metric]["valid_id"],
                test_id=metric_dict[metric]["test_id"],
                minimize=minimize_dict[metric],
                sample=args.sample,
                N=args.N,
                n_repeats=args.n_repeats
            )
            df_overtuning["metric"] = metric
            results.append(df_overtuning.reset_index(level=1, drop=True).reset_index())
            rel_pathology = df_overtuning[df_overtuning["final_iteration"] == True]["no_progress"].mean()
            rel_pathology_dict.update({metric: round(rel_pathology, 2)})
        results_df = pd.concat(results)
        results_df.to_csv("csvs/lcbench.csv", index=False)
    elif args.benchmark == "reshuffling":
        if args.reshuffling_resampling == "holdout":
            df = pd.read_csv("reshuffling/results_holdout.csv", usecols=["seed", "data_id", "resampling", "train_valid_size", "classifier", "optimizer", "metric", "valid", "test", "test_retrained"])
        elif args.reshuffling_resampling == "repeatedholdout":
            df = pd.read_csv("reshuffling/results_repeatedholdout.csv", usecols=["seed", "data_id", "resampling", "train_valid_size", "classifier", "optimizer", "metric", "valid", "test", "test_retrained", "test_ensemble"])
            df["test"] = df["test_ensemble"]
            df = df.loc[df["resampling"].isin(["cv_5_5_True", "cv_5_5_False", "repeatedholdout_02_5_True", "repeatedholdout_02_5_False"])]
        elif args.reshuffling_resampling == "cv":
            df = pd.read_csv("reshuffling/results_cv_postprocessed.csv", usecols=["seed", "data_id", "resampling", "train_valid_size", "classifier", "optimizer", "metric", "valid", "test", "test_retrained", "test_ensemble"])
            df["test"] = df["test_ensemble"]
        elif args.reshuffling_resampling == "cv_repeated":
            df = pd.read_csv("reshuffling/results_cv_repeated.csv", usecols=["seed", "data_id", "resampling", "train_valid_size", "classifier", "optimizer", "metric", "valid", "test", "test_retrained", "test_ensemble"])
            df["test"] = df["test_ensemble"]
        else:
            raise ValueError("Invalid reshuffling resampling method")
        results = []
        df["reshuffle"] = df["resampling"].str.contains("True")
        df = df[(df["train_valid_size"] == args.reshuffling_train_valid_size) & (df["reshuffle"] == args.reshuffling_reshuffled) & (df["classifier"] == args.reshuffling_classifier) & (df["optimizer"] == args.reshuffling_optimizer)]
        scenario_ids = ["seed", "data_id"]
        valid_id = "valid"
        if args.reshuffling_retrained:
            test_id = "test_retrained"
        else:
            test_id = "test"
        rel_pathology_dict = {}
        if args.reshuffling_optimizer == "random":
            metrics = ["accuracy", "auc", "balanced_accuracy", "logloss"]
        else:
            metrics = ["auc"]
        for metric in metrics:
            df_overtuning = df[df["metric"] == metric].groupby(scenario_ids).apply(
                compute_overtuning,
                valid_id=valid_id,
                test_id=test_id,
                minimize=True,  # collected data is already corrected for minimization
                sample=args.sample,
                N=args.N,
                n_repeats=args.n_repeats
            )
            df_overtuning["metric"] = metric
            df_overtuning["train_valid_size"] = args.reshuffling_train_valid_size
            df_overtuning["resampling"] = args.reshuffling_resampling
            df_overtuning["reshuffled"] = args.reshuffling_reshuffled
            df_overtuning["classifier"] = args.reshuffling_classifier
            df_overtuning["optimizer"] = args.reshuffling_optimizer
            df_overtuning["retrained"] = args.reshuffling_retrained
            results.append(df_overtuning.reset_index(level=2, drop=True).reset_index())
            rel_pathology = df_overtuning[df_overtuning["final_iteration"] == True]["no_progress"].mean()
            rel_pathology_dict.update({metric: round(rel_pathology, 2)})
        results_df = pd.concat(results)
        results_df.to_csv(f"csvs/{args.benchmark}_{args.reshuffling_train_valid_size}_{args.reshuffling_resampling}_{args.reshuffling_reshuffled}_{args.reshuffling_classifier}_{args.reshuffling_optimizer}_{args.reshuffling_retrained}.csv", index=False)
    elif args.benchmark == "wdtb":
        results = []
        df = pd.read_csv("wdtb/data.csv")
        df = df[df["model"] == args.wdtb_model]
        scenario_ids = ["dataset"]
        minimize_dict = {
            "accuracy": False,
            "r2": False
        }
        rel_pathology_dict = {}
        if args.wdtb_subbenchmark in ["categorical_classification_medium", "numerical_classification_medium"]:
            metric = "accuracy"
        elif args.wdtb_subbenchmark in ["categorical_regression_medium", "numerical_regression_medium"]:
            metric = "r2"
        else:
            raise ValueError("Invalid wdtb subbenchmark")
        df_overtuning = df[df["benchmark"] == args.wdtb_subbenchmark].groupby(scenario_ids).apply(
                compute_overtuning,
                valid_id="valid",
                test_id="test",
                minimize=minimize_dict[metric],
                sample=args.sample,
                N=args.N,
                n_repeats=args.n_repeats
            )
        df_overtuning["benchmark"] = args.wdtb_subbenchmark
        df_overtuning["metric"] = metric
        df_overtuning["model"] = args.wdtb_model
        results.append(df_overtuning.reset_index(level=1, drop=True).reset_index())
        rel_pathology = df_overtuning[df_overtuning["final_iteration"] == True]["no_progress"].mean()
        rel_pathology_dict.update({metric: round(rel_pathology, 2)})
        results_df = pd.concat(results)
        results_df.to_csv(f"csvs/{args.benchmark}_{args.wdtb_subbenchmark}_{args.wdtb_model}.csv", index=False)
    elif args.benchmark == "pd1":
        results = []
        df = pd.read_csv("pd1/data.csv")
        # subset to the maximum epoch rows for each task
        df = df[df["epoch"] == df.groupby("task")["epoch"].transform("max")]
        scenario_ids = ["task"]
        metric_dict = {
            "ce_loss": {"valid_id": "valid_ce_loss", "test_id": "test_ce_loss"},
            "error": {"valid_id": "valid_error", "test_id": "test_error"}
        }
        minimize_dict = {
            "ce_loss": True,
            "error": True
        }
        rel_pathology_dict = {}
        for metric in ["ce_loss", "error"]:
            df_overtuning = df.groupby(scenario_ids).apply(
                compute_overtuning,
                valid_id=metric_dict[metric]["valid_id"],
                test_id=metric_dict[metric]["test_id"],
                minimize=minimize_dict[metric],
                sample=args.sample,
                N=args.N,
                n_repeats=args.n_repeats
            )
            df_overtuning["metric"] = metric
            results.append(df_overtuning.reset_index(level=1, drop=True).reset_index())
            rel_pathology = df_overtuning[df_overtuning["final_iteration"] == True]["no_progress"].mean()
            rel_pathology_dict.update({metric: round(rel_pathology, 2)})
        results_df = pd.concat(results)
        results_df.to_csv(f"csvs/{args.benchmark}.csv", index=False)
    elif args.benchmark == "fcnet":
        results = []
        df_naval_propulsion = pd.read_csv("fcnet/data_naval_propulsion.csv")
        df_naval_propulsion = df_naval_propulsion[df_naval_propulsion["epoch"] == 100]
        df_naval_propulsion["task"] = "naval_propulsion"
        df_parkinsons_telemonitoring = pd.read_csv("fcnet/data_parkinsons_telemonitoring.csv")
        df_parkinsons_telemonitoring = df_parkinsons_telemonitoring[df_parkinsons_telemonitoring["epoch"] == 100]
        df_parkinsons_telemonitoring["task"] = "parkinsons_telemonitoring"
        df_protein_structure = pd.read_csv("fcnet/data_protein_structure.csv")
        df_protein_structure = df_protein_structure[df_protein_structure["epoch"] == 100]
        df_protein_structure["task"] = "protein_structure"
        df_slice_localization = pd.read_csv("fcnet/data_slice_localization.csv")
        df_slice_localization = df_slice_localization[df_slice_localization["epoch"] == 100]
        df_slice_localization["task"] = "slice_localization"
        df = pd.concat([df_naval_propulsion, df_parkinsons_telemonitoring, df_protein_structure, df_slice_localization])
        del df_naval_propulsion
        del df_parkinsons_telemonitoring
        del df_protein_structure
        del df_slice_localization

        scenario_ids = ["task", "repl"]
        metric_dict = {
            "mse": {"valid_id": "valid_mse", "test_id": "final_test_error"}
        }
        minimize_dict = {
            "mse": True
        }
        rel_pathology_dict = {}
        for metric in ["mse"]:
            df_overtuning = df.groupby(scenario_ids).apply(
                compute_overtuning,
                valid_id=metric_dict[metric]["valid_id"],
                test_id=metric_dict[metric]["test_id"],
                minimize=minimize_dict[metric],
                sample=args.sample,
                N=args.N,
                n_repeats=args.n_repeats
            )
            df_overtuning["metric"] = metric
            results.append(df_overtuning.reset_index(level=1, drop=True).reset_index())
            rel_pathology = df_overtuning[df_overtuning["final_iteration"] == True]["no_progress"].mean()
            rel_pathology_dict.update({metric: round(rel_pathology, 2)})
        results_df = pd.concat(results)
        results_df.to_csv(f"csvs/{args.benchmark}.csv", index=False)
    elif args.benchmark == "tabzilla":
        results = []
        df = pd.read_csv("tabzilla/data.csv")
        df = df[df["model"] == args.tabzilla_model]
        scenario_ids = ["dataset", "fold_id"]
        metric_dict = {
            "accuracy": {"valid_id": "accuracy_val", "test_id": "accuracy_test"},
            "auc": {"valid_id": "auc_val", "test_id": "auc_test"},
            "f1": {"valid_id": "f1_val", "test_id": "f1_test"},
            "logloss": {"valid_id": "log_loss_val", "test_id": "log_loss_test"},
        }
        minimize_dict = {
            "accuracy": False,
            "auc": False,
            "f1": False,
            "logloss": True,
        }
        rel_pathology_dict = {}
        for metric in ["accuracy", "auc", "f1", "logloss"]:
            df_overtuning = df[df["benchmark"] == args.tabzilla_subbenchmark].groupby(scenario_ids).apply(
                compute_overtuning,
                valid_id=metric_dict[metric]["valid_id"],
                test_id=metric_dict[metric]["test_id"],
                minimize=minimize_dict[metric],
                sample=args.sample,
                N=args.N,
                n_repeats=args.n_repeats
            )
            df_overtuning["benchmark"] = args.tabzilla_subbenchmark
            df_overtuning["metric"] = metric
            df_overtuning["model"] = args.tabzilla_model
            results.append(df_overtuning.reset_index(level=2, drop=True).reset_index())
            rel_pathology = df_overtuning[df_overtuning["final_iteration"] == True]["no_progress"].mean()
            rel_pathology_dict.update({metric: round(rel_pathology, 2)})
        results_df = pd.concat(results)
        results_df.to_csv(f"csvs/{args.benchmark}_{args.tabzilla_subbenchmark}_{args.tabzilla_model}.csv", index=False)
    elif args.benchmark == "tabrepo":
        results = []
        df = pd.read_csv("tabrepo/data.csv")
        df = df[df["model"] == args.tabrepo_model]
        scenario_ids = ["dataset", "fold"]
        minimize_dict = {
            "log_loss": True,
            "rmse": True,
            "roc_auc": True
        }
        rel_pathology_dict = {}
        if args.tabrepo_subbenchmark == "binary":
            metric = "roc_auc"
        elif args.tabrepo_subbenchmark == "multiclass":
            metric = "log_loss"
        elif args.tabrepo_subbenchmark == "regression":
            metric = "rmse"
        else:
            raise ValueError("Invalid tabrepo subbenchmark")
        df_overtuning = df[df["problem_type"] == args.tabrepo_subbenchmark].groupby(scenario_ids).apply(
            compute_overtuning,
            valid_id="valid",
            test_id="test",
            minimize=minimize_dict[metric],
            sample=args.sample,
            N=args.N,
            n_repeats=args.n_repeats
        )
        df_overtuning["benchmark"] = args.tabrepo_subbenchmark
        df_overtuning["metric"] = metric
        df_overtuning["model"] = args.tabrepo_model
        results.append(df_overtuning.reset_index(level=2, drop=True).reset_index())
        rel_pathology = df_overtuning[df_overtuning["final_iteration"] == True]["no_progress"].mean()
        rel_pathology_dict.update({metric: round(rel_pathology, 2)})
        results_df = pd.concat(results)
        results_df.to_csv(f"csvs/{args.benchmark}_{args.tabrepo_subbenchmark}_{args.tabrepo_model}.csv", index=False)

