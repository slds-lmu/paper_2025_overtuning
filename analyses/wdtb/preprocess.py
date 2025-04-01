if __name__ == "__main__":
    # FIXME: add HPCs and split data
    # FIXME: clean up benchmark, dataset and model names
    import pandas as pd

    data = pd.read_csv("tabular_benchmark.csv")
    benchmarks = data["benchmark"].unique()

    data.loc[data["model_name"] == "FT Transformer", "model_name"] = "FT_Transformer"

    data_list = []

    for benchmark in benchmarks:
        tmp_benchmark = data[data["benchmark"] == benchmark]
        models = tmp_benchmark["model_name"].unique()
        for model in models:
            tmp_model = tmp_benchmark[(tmp_benchmark["model_name"] == model) & (tmp_benchmark["hp"] == "random")]
            datasets = tmp_model["data__keyword"].unique()
            for dataset in datasets:
                tmp_data = tmp_model[(tmp_model["data__keyword"] == dataset)]
                tmp_data = tmp_data[["data__keyword", "benchmark", "model_name", "mean_train_score", "mean_val_score", "mean_test_score"]]
                tmp_data = tmp_data.rename({"data__keyword": "dataset", "benchmark": "benchmark", "model_name": "model", "mean_train_score": "train", "mean_val_score": "valid", "mean_test_score": "test"}, axis=1)
                tmp_data = tmp_data.reset_index(drop=True)
                tmp_data["iteration"] = range(1, len(tmp_data) + 1)
                if "classification" in benchmark:
                    tmp_data["metric"] = "accuracy"
                else:
                    tmp_data["metric"] = "r2"
                data_list.append(tmp_data)

    df = pd.concat(data_list, axis=0)
    df.to_csv("data.csv", index=False)
