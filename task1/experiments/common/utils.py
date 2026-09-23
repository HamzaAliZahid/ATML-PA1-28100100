import pandas as pd
import os
import numpy as np

SEED = 6304

def save_results(model, experiment, accuracy, macro_f1, mean_confidence):
    results = pd.DataFrame([{
        "model": model,
        "experiment": experiment,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mean_confidence": mean_confidence
    }])

    results.to_csv(
        "task1/results/task1_results.csv",
        mode="a",
        header=not os.path.exists("../results/task1_results.csv"),
        index=False
    )

def select_test_indices(labels):
    labels = np.array(labels)

    rng = np.random.default_rng(SEED)

    test_indices = []

    for class_id in range(10):
        class_indices = np.where(labels == class_id)[0]

        selected = rng.choice(
            class_indices,
            size=50,
            replace=False
        )

        test_indices.extend(selected)

    return np.array(test_indices)