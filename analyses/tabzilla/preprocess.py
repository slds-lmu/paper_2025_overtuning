if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv("metadataset_clean.csv")
    benchmarks = ["binary", "multiclass"]
    rename_dict = {
        "dataset_fold_id": "dataset_fold_id",
        "dataset_name": "dataset",
        "target_type": "benchmark",
        "alg_name": "model",
        "Log Loss__train": "log_loss_train",
        "Log Loss__val": "log_loss_val",
        "Log Loss__test": "log_loss_test",
        "AUC__train": "auc_train",
        "AUC__val": "auc_val",
        "AUC__test": "auc_test",
        "Accuracy__train": "accuracy_train",
        "Accuracy__val": "accuracy_val",
        "Accuracy__test": "accuracy_test",
        "F1__train": "f1_train",
        "F1__val": "f1_val",
        "F1__test": "f1_test",
        "training_time": "time_train",
        "eval-time__train": "time_predict_train",
        "eval-time__val": "time_predict_val",
        "eval-time__test": "time_predict_test",
    }
    data["fold_id"] = data["dataset_fold_id"].apply(lambda x: int(x.split("_")[-1]))
    data.loc[data["target_type"] == "classification", "target_type"] = "multiclass"
    data = data.rename(rename_dict, axis=1)

    data_list = []

    for benchmark in benchmarks:
        tmp_benchmark = data[data["benchmark"] == benchmark]
        models = tmp_benchmark["model"].unique()
        for model in models:
            tmp_model = tmp_benchmark[(tmp_benchmark["model"] == model) & (tmp_benchmark["hparam_source"] != "default")]
            datasets_fold_ids = tmp_model["dataset_fold_id"].unique()
            for dataset_fold_id in datasets_fold_ids:
                tmp_data = tmp_model[(tmp_model["dataset_fold_id"] == dataset_fold_id)]
                tmp_data = tmp_data[["benchmark", "dataset", "fold_id", "model", "fold_id", "log_loss_train", "log_loss_val", "log_loss_test", "auc_train", "auc_val", "auc_test", "accuracy_train", "accuracy_val", "accuracy_test", "f1_train", "f1_val", "f1_test", "time_train", "time_predict_train", "time_predict_val", "time_predict_test"]]
                tmp_data = tmp_data.drop_duplicates()
                tmp_data = tmp_data.reset_index(drop=True)
                tmp_data["iteration"] = range(1, len(tmp_data) + 1)
                any_NAs = pd.isna(tmp_data).any().any()
                if len(tmp_data) == 29 and not any_NAs:
                    data_list.append(tmp_data)

    df = pd.concat(data_list, axis=0)
    df.to_csv("data.csv", index=False)
