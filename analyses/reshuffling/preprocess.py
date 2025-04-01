if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    results_cv = pd.read_csv("results_cv.csv")
    optimizer = "hebo_makarova"

    results_cv_relevant = results_cv[results_cv["optimizer"] == optimizer].copy()
    results_cv_rest = results_cv[results_cv["optimizer"] != optimizer]
    del results_cv

    group_keys = [
        "data_id", "seed", "classifier",
        "train_valid_size", "resampling", "metric"
    ]

    def truncate_group(df: pd.DataFrame):
        assert len(df) == 250
        early_triggered = df["early_stopping_triggered"].values
        if early_triggered.any():
            idx = np.argmax(early_triggered)  # first True
            truncated = df.iloc[:idx].copy()
            if len(truncated) > 0:
                truncated.iloc[-1, df.columns.get_loc("early_stopping_triggered")] = True
            return truncated
        return df

    results_cv_relevant = (
        results_cv_relevant
        .groupby(group_keys, group_keys=False, sort=False)
        .apply(truncate_group)
        .reset_index(drop=True)
    )

    results_cv_final = pd.concat([results_cv_relevant, results_cv_rest])
    results_cv_final.to_csv("results_cv_postprocessed.csv", index=False)
