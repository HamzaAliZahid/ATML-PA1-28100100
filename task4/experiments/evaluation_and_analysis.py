import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve

RESULTS_DIR = "task4/results"
FIGURES_DIR = "task4/results/figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def msp_score(logits):
    probs = softmax(logits)
    return 1.0 - np.max(probs, axis=1)

def mls_score(logits):
    return -np.max(logits, axis=1)

def energy_score(logits):
    max_logits = np.max(logits, axis=1, keepdims=True)

    return -(
        max_logits[:, 0]
        + np.log(
            np.sum(
                np.exp(logits - max_logits),
                axis=1
            )
        )
    )


def mahalanobis_score(features, class_means, variances):
    scores = []

    for x in features:

        distances = []

        for mean in class_means:

            distance = np.sum(
                ((x - mean) ** 2) / variances
            )

            distances.append(distance)

        scores.append(
            np.min(distances)
        )

    return np.asarray(scores)

def compute_mahalanobis_statistics(
    train_features,
    train_labels
):

    class_means = []

    for class_id in range(10):

        class_features = train_features[
            train_labels == class_id
        ]

        mean = np.mean(
            class_features,
            axis=0
        )

        class_means.append(mean)

    class_means = np.asarray(
        class_means
    )

    variances = np.var(
        train_features,
        axis=0
    )

    variances = variances + 1e-6

    return class_means, variances

def get_threshold(validation_scores):
    return np.percentile(
        validation_scores,
        95
    )

def acceptance_rate(
    scores,
    threshold
):

    return np.mean(
        scores <= threshold
    )


def rejection_rate(
    scores,
    threshold
):

    return np.mean(
        scores > threshold
    )

def compute_auroc(
    known_scores,
    unknown_scores
):
    y_true = np.concatenate([
        np.zeros(len(known_scores)),
        np.ones(len(unknown_scores))
    ])

    scores = np.concatenate([
        known_scores,
        unknown_scores
    ])

    return roc_auc_score(
        y_true,
        scores
    )


def evaluate_score(
    validation_scores,
    known_test_scores,
    near_scores,
    far_scores
):

    threshold = get_threshold(
        validation_scores
    )

    all_unknown_scores = np.concatenate([
        near_scores,
        far_scores
    ])

    result = {

        "threshold":
            threshold,

        "known_acceptance":
            acceptance_rate(
                known_test_scores,
                threshold
            ),

        "near_rejection":
            rejection_rate(
                near_scores,
                threshold
            ),

        "far_rejection":
            rejection_rate(
                far_scores,
                threshold
            ),

        "near_auroc":
            compute_auroc(
                known_test_scores,
                near_scores
            ),

        "far_auroc":
            compute_auroc(
                known_test_scores,
                far_scores
            ),

        "all_auroc":
            compute_auroc(
                known_test_scores,
                all_unknown_scores
            )
    }

    return result

def result_to_row(
    model_name,
    score_name,
    result
):

    return {

        "Model":
            model_name,

        "Score":
            score_name,

        "Threshold":
            result["threshold"],

        "Known Acceptance (%)":
            result["known_acceptance"] * 100,

        "Near Rejection (%)":
            result["near_rejection"] * 100,

        "Far Rejection (%)":
            result["far_rejection"] * 100,

        "Near AUROC":
            result["near_auroc"],

        "Far AUROC":
            result["far_auroc"],

        "All Unknown AUROC":
            result["all_auroc"]
    }


def print_result(
    model_name,
    score_name,
    result
):
    print("=" * 60)
    print(model_name, "-", score_name)
    print("=" * 60)

    print(
        f"Threshold: "
        f"{result['threshold']:.6f}"
    )

    print(
        f"Known acceptance: "
        f"{result['known_acceptance'] * 100:.2f}%"
    )

    print(
        f"Near rejection: "
        f"{result['near_rejection'] * 100:.2f}%"
    )

    print(
        f"Far rejection: "
        f"{result['far_rejection'] * 100:.2f}%"
    )

    print(
        f"Near AUROC: "
        f"{result['near_auroc']:.4f}"
    )

    print(
        f"Far AUROC: "
        f"{result['far_auroc']:.4f}"
    )

    print(
        f"All unknown AUROC: "
        f"{result['all_auroc']:.4f}"
    )

def get_roc_data(
    known_scores,
    unknown_scores
):

    y_true = np.concatenate([
        np.zeros(len(known_scores)),
        np.ones(len(unknown_scores))
    ])

    scores = np.concatenate([
        known_scores,
        unknown_scores
    ])

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores
    )

    auroc = roc_auc_score(
        y_true,
        scores
    )

    return fpr, tpr, auroc

def get_false_accepts(
    unknown_scores,
    unknown_classes,
    predicted_classes,
    threshold,
    n=3
):
    unknown_scores = np.asarray(
        unknown_scores
    )

    accepted_indices = np.where(
        unknown_scores <= threshold
    )[0]

    if len(accepted_indices) == 0:
        return []

    accepted_indices = accepted_indices[
        np.argsort(
            unknown_scores[
                accepted_indices
            ]
        )
    ]

    accepted_indices = accepted_indices[:n]

    failures = []

    for index in accepted_indices:

        failures.append({

            "index":
                int(index),

            "unknown_class":
                unknown_classes[index],

            "predicted_class":
                predicted_classes[index],

            "score":
                float(unknown_scores[index]),

            "threshold":
                float(threshold)
        })

    return failures


def print_failures(
    title,
    failures
):
    print("=" * 60)
    print(title)
    print("=" * 60)

    if len(failures) == 0:

        print(
            "No incorrectly accepted examples found."
        )

        return

    for i, failure in enumerate(
        failures,
        start=1
    ):

        print(
            f"Failure {i}"
        )

        print(
            f"Index: "
            f"{failure['index']}"
        )

        print(
            f"Unknown class: "
            f"{failure['unknown_class']}"
        )

        print(
            f"Predicted CIFAR-10 class: "
            f"{failure['predicted_class']}"
        )

        print(
            f"MLS score: "
            f"{failure['score']:.6f}"
        )

        print(
            f"Threshold: "
            f"{failure['threshold']:.6f}"
        )


def evaluate_model_mls(
    model_name,
    validation_logits,
    known_test_logits,
    near_logits,
    far_logits
):

    validation_scores = mls_score(
        validation_logits
    )

    known_scores = mls_score(
        known_test_logits
    )

    near_scores = mls_score(
        near_logits
    )

    far_scores = mls_score(
        far_logits
    )

    result = evaluate_score(
        validation_scores,
        known_scores,
        near_scores,
        far_scores
    )

    print_result(
        model_name,
        "MLS",
        result
    )

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("TASK 4 - PART 6")
    print("COMMON EVALUATION AND FAILURE ANALYSIS")
    print("=" * 70)

    validation_logits = np.load(
        os.path.join(
            RESULTS_DIR,
            "vanilla_validation_logits.npy"
        )
    )

    known_test_logits = np.load(
        os.path.join(
            RESULTS_DIR,
            "vanilla_known_test_logits.npy"
        )
    )

    near_logits = np.load(
        os.path.join(
            RESULTS_DIR,
            "vanilla_near_logits.npy"
        )
    )

    far_logits = np.load(
        os.path.join(
            RESULTS_DIR,
            "vanilla_far_logits.npy"
        )
    )


    validation_features = np.load(
        os.path.join(
            RESULTS_DIR,
            "vanilla_validation_features.npy"
        )
    )

    known_test_features = np.load(
        os.path.join(
            RESULTS_DIR,
            "vanilla_known_test_features.npy"
        )
    )

    near_features = np.load(
        os.path.join(
            RESULTS_DIR,
            "vanilla_near_features.npy"
        )
    )

    far_features = np.load(
        os.path.join(
            RESULTS_DIR,
            "vanilla_far_features.npy"
        )
    )


    train_features = np.load(
        os.path.join(
            RESULTS_DIR,
            "vanilla_train_features.npy"
        )
    )

    train_labels = np.load(
        os.path.join(
            RESULTS_DIR,
            "vanilla_train_labels.npy"
        )
    )

    print("Vanilla output shapes:")

    print(
        "Validation logits:",
        validation_logits.shape
    )

    print(
        "Known test logits:",
        known_test_logits.shape
    )

    print(
        "Near logits:",
        near_logits.shape
    )

    print(
        "Far logits:",
        far_logits.shape
    )

    print(
        "Validation features:",
        validation_features.shape
    )

    print(
        "Known test features:",
        known_test_features.shape
    )

    print(
        "Near features:",
        near_features.shape
    )

    print(
        "Far features:",
        far_features.shape
    )

    class_means, variances = (
        compute_mahalanobis_statistics(
            train_features,
            train_labels
        )
    )

    validation_msp = msp_score(
        validation_logits
    )

    known_msp = msp_score(
        known_test_logits
    )

    near_msp = msp_score(
        near_logits
    )

    far_msp = msp_score(
        far_logits
    )


    vanilla_msp_result = evaluate_score(
        validation_msp,
        known_msp,
        near_msp,
        far_msp
    )

    print_result(
        "Vanilla",
        "MSP",
        vanilla_msp_result
    )

    validation_mls = mls_score(
        validation_logits
    )

    known_mls = mls_score(
        known_test_logits
    )

    near_mls = mls_score(
        near_logits
    )

    far_mls = mls_score(
        far_logits
    )


    vanilla_mls_result = evaluate_score(
        validation_mls,
        known_mls,
        near_mls,
        far_mls
    )

    print_result(
        "Vanilla",
        "MLS",
        vanilla_mls_result
    )

    validation_energy = energy_score(
        validation_logits
    )

    known_energy = energy_score(
        known_test_logits
    )

    near_energy = energy_score(
        near_logits
    )

    far_energy = energy_score(
        far_logits
    )


    vanilla_energy_result = evaluate_score(
        validation_energy,
        known_energy,
        near_energy,
        far_energy
    )

    print_result(
        "Vanilla",
        "Energy",
        vanilla_energy_result
    )

    validation_mah = mahalanobis_score(
        validation_features,
        class_means,
        variances
    )

    known_mah = mahalanobis_score(
        known_test_features,
        class_means,
        variances
    )

    near_mah = mahalanobis_score(
        near_features,
        class_means,
        variances
    )

    far_mah = mahalanobis_score(
        far_features,
        class_means,
        variances
    )


    vanilla_mah_result = evaluate_score(
        validation_mah,
        known_mah,
        near_mah,
        far_mah
    )

    print_result(
        "Vanilla",
        "Mahalanobis",
        vanilla_mah_result
    )

    table1 = [

        result_to_row(
            "Vanilla",
            "MSP",
            vanilla_msp_result
        ),

        result_to_row(
            "Vanilla",
            "MLS",
            vanilla_mls_result
        ),

        result_to_row(
            "Vanilla",
            "Energy",
            vanilla_energy_result
        ),

        result_to_row(
            "Vanilla",
            "Mahalanobis",
            vanilla_mah_result
        )
    ]


    table1 = pd.DataFrame(
        table1
    )

    print("=" * 70)
    print("TABLE 1 - VANILLA SCORE COMPARISON")
    print("=" * 70)

    print(
        table1.to_string(
            index=False
        )
    )


    table1.to_csv(
        os.path.join(
            RESULTS_DIR,
            "part6_table1_scores.csv"
        ),
        index=False
    )

    all_unknown_msp = np.concatenate([
        near_msp,
        far_msp
    ])

    all_unknown_mls = np.concatenate([
        near_mls,
        far_mls
    ])

    all_unknown_mah = np.concatenate([
        near_mah,
        far_mah
    ])


    msp_fpr, msp_tpr, msp_auc = get_roc_data(
        known_msp,
        all_unknown_msp
    )

    mls_fpr, mls_tpr, mls_auc = get_roc_data(
        known_mls,
        all_unknown_mls
    )

    mah_fpr, mah_tpr, mah_auc = get_roc_data(
        known_mah,
        all_unknown_mah
    )


    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5)
    )


    axes[0].plot(
        msp_fpr,
        msp_tpr
    )

    axes[0].plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    axes[0].set_title(
        f"MSP (AUROC={msp_auc:.3f})"
    )

    axes[0].set_xlabel(
        "False Positive Rate"
    )

    axes[0].set_ylabel(
        "True Positive Rate"
    )


    axes[1].plot(
        mls_fpr,
        mls_tpr
    )

    axes[1].plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    axes[1].set_title(
        f"MLS (AUROC={mls_auc:.3f})"
    )

    axes[1].set_xlabel(
        "False Positive Rate"
    )

    axes[1].set_ylabel(
        "True Positive Rate"
    )


    axes[2].plot(
        mah_fpr,
        mah_tpr
    )

    axes[2].plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    axes[2].set_title(
        f"Mahalanobis (AUROC={mah_auc:.3f})"
    )

    axes[2].set_xlabel(
        "False Positive Rate"
    )

    axes[2].set_ylabel(
        "True Positive Rate"
    )


    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            "part6_roc_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    gcsc_validation_logits = np.load(
        os.path.join(
            RESULTS_DIR,
            "gcsc_validation_logits.npy"
        )
    )

    gcsc_known_test_logits = np.load(
        os.path.join(
            RESULTS_DIR,
            "gcsc_known_test_logits.npy"
        )
    )

    gcsc_near_logits = np.load(
        os.path.join(
            RESULTS_DIR,
            "gcsc_near_logits.npy"
        )
    )

    gcsc_far_logits = np.load(
        os.path.join(
            RESULTS_DIR,
            "gcsc_far_logits.npy"
        )
    )

    gcsc_validation_mls = mls_score(
        gcsc_validation_logits
    )

    gcsc_known_mls = mls_score(
        gcsc_known_test_logits
    )

    gcsc_near_mls = mls_score(
        gcsc_near_logits
    )

    gcsc_far_mls = mls_score(
        gcsc_far_logits
    )


    gcsc_result = evaluate_score(
        gcsc_validation_mls,
        gcsc_known_mls,
        gcsc_near_mls,
        gcsc_far_mls
    )


    print_result(
        "GCSC",
        "MLS",
        gcsc_result
    )

    proser_validation_logits = np.load(
        os.path.join(
            RESULTS_DIR,
            "proser_validation_logits.npy"
        )
    )

    proser_known_test_logits = np.load(
        os.path.join(
            RESULTS_DIR,
            "proser_known_test_logits.npy"
        )
    )

    proser_near_logits = np.load(
        os.path.join(
            RESULTS_DIR,
            "proser_near_logits.npy"
        )
    )

    proser_far_logits = np.load(
        os.path.join(
            RESULTS_DIR,
            "proser_far_logits.npy"
        )
    )

    proser_validation_mls = mls_score(
        proser_validation_logits[:, :10]
    )

    proser_known_mls = mls_score(
        proser_known_test_logits[:, :10]
    )

    proser_near_mls = mls_score(
        proser_near_logits[:, :10]
    )

    proser_far_mls = mls_score(
        proser_far_logits[:, :10]
    )


    proser_mls_result = evaluate_score(
        proser_validation_mls,
        proser_known_mls,
        proser_near_mls,
        proser_far_mls
    )


    print_result(
        "PROSER",
        "MLS",
        proser_mls_result
    )

    proser_validation_placeholder = np.load(
        os.path.join(
            RESULTS_DIR,
            "proser_validation_placeholder_score.npy"
        )
    )

    proser_known_placeholder = np.load(
        os.path.join(
            RESULTS_DIR,
            "proser_known_placeholder_score.npy"
        )
    )

    proser_near_placeholder = np.load(
        os.path.join(
            RESULTS_DIR,
            "proser_near_placeholder_score.npy"
        )
    )

    proser_far_placeholder = np.load(
        os.path.join(
            RESULTS_DIR,
            "proser_far_placeholder_score.npy"
        )
    )


    proser_placeholder_result = evaluate_score(
        proser_validation_placeholder,
        proser_known_placeholder,
        proser_near_placeholder,
        proser_far_placeholder
    )


    print_result(
        "PROSER",
        "Placeholder",
        proser_placeholder_result
    )

    table2 = [

        result_to_row(
            "Vanilla",
            "MLS",
            vanilla_mls_result
        ),

        result_to_row(
            "GCSC",
            "MLS",
            gcsc_result
        ),

        result_to_row(
            "PROSER",
            "MLS",
            proser_mls_result
        ),

        result_to_row(
            "PROSER",
            "Placeholder",
            proser_placeholder_result
        )
    ]


    table2 = pd.DataFrame(
        table2
    )

    print("=" * 70)
    print("TABLE 2 - TRAINED MODEL COMPARISON")
    print("=" * 70)

    print(
        table2.to_string(
            index=False
        )
    )


    table2.to_csv(
        os.path.join(
            RESULTS_DIR,
            "part6_table2_models.csv"
        ),
        index=False
    )

    vanilla_mls_threshold = (
        vanilla_mls_result["threshold"]
    )

    near_true_classes = np.load(
        os.path.join(
            RESULTS_DIR,
            "near_true_classes.npy"
        ),
        allow_pickle=True
    )

    near_predicted_classes = np.load(
        os.path.join(
            RESULTS_DIR,
            "near_predicted_classes.npy"
        ),
        allow_pickle=True
    )

    far_true_classes = np.load(
        os.path.join(
            RESULTS_DIR,
            "far_true_classes.npy"
        ),
        allow_pickle=True
    )

    far_predicted_classes = np.load(
        os.path.join(
            RESULTS_DIR,
            "far_predicted_classes.npy"
        ),
        allow_pickle=True
    )

    near_failures = get_false_accepts(
        near_mls,
        near_true_classes,
        near_predicted_classes,
        vanilla_mls_threshold,
        n=3
    )

    far_failures = get_false_accepts(
        far_mls,
        far_true_classes,
        far_predicted_classes,
        vanilla_mls_threshold,
        n=3
    )


    print_failures(
        "VANILLA MLS - NEAR UNKNOWN FAILURES",
        near_failures
    )


    print_failures(
        "VANILLA MLS - FAR UNKNOWN FAILURES",
        far_failures
    )

    failure_rows = []


    for failure in near_failures:

        failure_rows.append({

            "Group":
                "Near",

            "Index":
                failure["index"],

            "Unknown Class":
                failure["unknown_class"],

            "Predicted CIFAR-10 Class":
                failure["predicted_class"],

            "MLS Score":
                failure["score"],

            "Threshold":
                failure["threshold"]
        })


    for failure in far_failures:

        failure_rows.append({

            "Group":
                "Far",

            "Index":
                failure["index"],

            "Unknown Class":
                failure["unknown_class"],

            "Predicted CIFAR-10 Class":
                failure["predicted_class"],

            "MLS Score":
                failure["score"],

            "Threshold":
                failure["threshold"]
        })


    failure_df = pd.DataFrame(
        failure_rows
    )


    failure_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "part6_failure_analysis.csv"
        ),
        index=False
    )

    print("=" * 70)
    print("TASK 4 PART 6 COMPLETE")
    print("=" * 70)

    print(
        "Saved:"
    )

    print(
        "- part6_table1_scores.csv"
    )

    print(
        "- part6_table2_models.csv"
    )

    print(
        "- part6_roc_comparison.png"
    )

    print(
        "- part6_failure_analysis.csv"
    )

    print(
        "Vanilla MLS threshold:",
        vanilla_mls_threshold
    )