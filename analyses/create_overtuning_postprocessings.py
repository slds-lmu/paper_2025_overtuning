if __name__ == "__main__":
    import os

    benchmark = "reshuffling"
    reshuffling_optimizers = ["random", "hebo", "smac", "hebo_makarova"]
    reshuffling_resamplings = ["holdout", "repeatedholdout", "cv", "cv_repeated"]
    reshuffling_train_valid_sizes = [500, 1000, 5000]
    reshuffling_reshuffles = [True, False]
    reshuffling_retraineds = [True, False]
    reshuffling_classifiers = ["catboost", "xgboost", "funnel_mlp", "logreg"]

    file = f"run_overtuning_postprocessing_{benchmark}.sh"
    with open(file, "+w") as f:
        for reshuffling_optimizer in reshuffling_optimizers:
            if reshuffling_optimizer == "hebo_makarova":
                reshuffling_resamplings_tmp = ["cv"]
            else:
                reshuffling_resamplings_tmp = reshuffling_resamplings
            for reshuffling_resampling in reshuffling_resamplings_tmp:
                for reshuffling_train_valid_size in reshuffling_train_valid_sizes:
                    for reshuffling_reshuffle in reshuffling_reshuffles:
                        for reshuffling_retrained in reshuffling_retraineds:
                            for reshuffling_classifier in reshuffling_classifiers:
                                f.write(
                                    f"python3 overtuning_postprocessing.py --benchmark {benchmark} --reshuffling_train_valid_size {reshuffling_train_valid_size} --reshuffling_resampling {reshuffling_resampling} --reshuffling_reshuffled {reshuffling_reshuffle} --reshuffling_classifier {reshuffling_classifier} --reshuffling_optimizer {reshuffling_optimizer} --reshuffling_retrained {reshuffling_retrained}\n"
                                )
    os.chmod(file, 0o755)

    benchmark = "wdtb"
    wdtb_models = ["RandomForest", "XGBoost", "GradientBoostingTree", "Resnet", "FT_Transformer", "SAINT", "MLP", "HistGradientBoostingTree"]
    wdtb_subbenchmarks = ["categorical_classification_medium", "numerical_classification_medium", "categorical_regression_medium", "numerical_regression_medium"]

    file = f"run_overtuning_postprocessing_{benchmark}.sh"
    with open(file, "+w") as f:
        for wdtb_subbenchmark in wdtb_subbenchmarks:
            for wdtb_model in wdtb_models:
                if wdtb_model == "HistGradientBoostingTree" and wdtb_subbenchmark == "numerical_classification_medium":
                    continue
                f.write(
                    f"python3 overtuning_postprocessing.py --benchmark {benchmark} --wdtb_subbenchmark {wdtb_subbenchmark} --wdtb_model {wdtb_model}\n"
                )
    os.chmod(file, 0o755)

    benchmark = "tabzilla"
    tabzilla_models = ["CatBoost", "DecisionTree", "DeepFM", "KNN", "LightGBM", "LinearModel", "MLP", "RandomForest", "STG", "SVM", "TabNet", "TabTransformer", "VIME", "XGBoost", "rtdl_MLP", "rtdl_ResNet", "DANet", "NAM", "NODE", "SAINT", "rtdl_FTTransformer", "TabPFNModel"]
    tabzilla_subbenchmarks = ["binary", "multiclass"]

    file = f"run_overtuning_postprocessing_{benchmark}.sh"
    with open(file, "+w") as f:
        for tabzilla_subbenchmark in tabzilla_subbenchmarks:
            for tabzilla_model in tabzilla_models:
                if tabzilla_model in ["DeepFM", "NAM"] and tabzilla_subbenchmark == "multiclass":
                    continue
                f.write(
                    f"python3 overtuning_postprocessing.py --benchmark {benchmark} --tabzilla_subbenchmark {tabzilla_subbenchmark} --tabzilla_model {tabzilla_model}\n"
                )
    os.chmod(file, 0o755)

    benchmark = "tabrepo"
    tabrepo_models = ["CatBoost", "ExtraTrees", "KNeighbors", "LightGBM", "LinearModel", "NeuralNetFastAI", "NeuralNetTorch", "RandomForest", "XGBoost"]
    tabrepo_subbenchmarks = ["binary", "multiclass", "regression"]

    file = f"run_overtuning_postprocessing_{benchmark}.sh"
    with open(file, "+w") as f:
        for tabrepo_subbenchmark in tabrepo_subbenchmarks:
            for tabrepo_model in tabrepo_models:
                f.write(
                    f"python3 overtuning_postprocessing.py --benchmark {benchmark} --tabrepo_subbenchmark {tabrepo_subbenchmark} --tabrepo_model {tabrepo_model}\n"
                )
    os.chmod(file, 0o755)

    benchmark = "lcbench"

    file = f"run_overtuning_postprocessing_{benchmark}.sh"
    with open(file, "+w") as f:
        f.write(
            f"python3 overtuning_postprocessing.py --benchmark {benchmark}\n"
        )
    os.chmod(file, 0o755)

    benchmark = "pd1"

    file = f"run_overtuning_postprocessing_{benchmark}.sh"
    with open(file, "+w") as f:
        f.write(
            f"python3 overtuning_postprocessing.py --benchmark {benchmark}\n"
        )
    os.chmod(file, 0o755)

    benchmark = "fcnet"

    file = f"run_overtuning_postprocessing_{benchmark}.sh"
    with open(file, "+w") as f:
        f.write(
            f"python3 overtuning_postprocessing.py --benchmark {benchmark}\n"
        )
    os.chmod(file, 0o755)
    
