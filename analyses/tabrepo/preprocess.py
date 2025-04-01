if __name__ == "__main__":
    # FIXME: add HPCs
    from tabrepo import load_repository, get_context, list_contexts, EvaluationRepository
    import numpy as np
    import pandas as pd

    context_name = "D244_F3_C1530_175"
    context = get_context(name=context_name)

    repo: EvaluationRepository = load_repository(context_name, cache=True)
    datasets = repo.datasets()
    data_list = []

    for dataset in datasets:
        dataset_info = repo.dataset_info(dataset=dataset)
        configs = repo.configs(datasets=[dataset], union=False)
        models = np.unique([config.split("_")[0] for config in configs]).tolist()
        configs_per_model = {}
        for model in models:
            configs_per_model[model] = [config for config in configs if config.startswith(model)]
            if len(configs_per_model[model]) < 30:
                del configs_per_model[model]
        models = list(configs_per_model.keys())
        for model in models:
            tmp_data = repo.metrics(datasets=[dataset], configs=configs_per_model[model])
            tmp_data.reset_index(inplace=True)
            tmp_data["dataset"] = dataset
            tmp_data["metric"] = dataset_info["metric"]
            tmp_data["problem_type"] = dataset_info["problem_type"]
            tmp_data["model"] = model
            tmp_data["iteration"] = range(1, len(tmp_data) + 1)
            tmp_data = tmp_data.rename({"metric_error": "test", "metric_error_val": "valid"}, axis=1)
            data_list.append(tmp_data)

    df = pd.concat(data_list, axis=0)
    df.to_csv("data.csv", index=False)
