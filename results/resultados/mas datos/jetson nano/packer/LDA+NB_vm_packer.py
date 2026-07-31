from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from collections import Counter

from resource_metrics import (
    base_run_metadata, benchmark_inference, measure_call,
    save_metrics, total_file_size_bytes
)

# Names of malware types in order of classes
malware_names = ["not_packed", "packed"]
# Function to load and normalize data
def load_data(file):
    data = []
    labels = []
    label_counter = Counter()
    label_classes = {}  # Dictionary for mapping one-hot labels to classes

    with open(file, 'r') as f:
        for i, line in enumerate(f):
            elements = line.strip().split(',')
            vector = list(map(int, elements[:1000]))
            one_hot_label = tuple(map(float, elements[1000:]))  # Convert to tuple for use in the counter
            
            if one_hot_label not in label_classes:
                label_classes[one_hot_label] = f"Class_{len(label_classes)}"  # Assign name to the class

            data.append(vector)
            labels.append(one_hot_label)
            label_counter[one_hot_label] += 1

    # Show the total number of samples per label
    print("\nLabel distribution:")
    for label, count in label_counter.items():
        print(f"{label}: {count}")

    return np.array(data), np.array(labels)

# Load data and split before fitting any preprocessing parameters
X, y_one_hot = load_data('one-hot-encoding1000.txt')
y = np.argmax(y_one_hot, axis=1)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
n_classes = len(np.unique(y_train))
lda = LDA(n_components=min(X_train_raw.shape[1], n_classes - 1))

def fit_preprocessing():
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)
    return X_train_lda, X_test_lda

(X_train, X_test), preprocessing_measure = measure_call(fit_preprocessing)
nb_model = GaussianNB()
_, fitting_measure = measure_call(lambda: nb_model.fit(X_train, y_train))
y_pred = nb_model.predict(X_test)

# Probabilities needed to calculate ROC-AUC and PR-AUC
y_score = nb_model.predict_proba(X_test)

# ROC-AUC and PR-AUC calculation
if n_classes == 2:
    # Positive class: class with index 1
    y_score_positive = y_score[:, 1]

    roc_auc = roc_auc_score(y_test, y_score_positive)
    pr_auc = average_precision_score(y_test, y_score_positive)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_score_positive)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Naive Bayes (ROC-AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve_LDA+NB1000.png")
    print("ROC curve saved as 'roc_curve_LDA+NB1000.png'")
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_score_positive)
    positive_prevalence = np.mean(y_test)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"Naive Bayes (PR-AUC = {pr_auc:.3f})")
    plt.axhline(
        y=positive_prevalence,
        linestyle="--",
        label=f"Baseline = {positive_prevalence:.3f}"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pr_curve_LDA+NB1000.png")
    print("Precision-Recall curve saved as 'pr_curve_LDA+NB1000.png'")
    plt.close()

else:
    # One-vs-Rest binarization for multiclass evaluation
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

    roc_auc = roc_auc_score(
        y_test_bin,
        y_score,
        multi_class="ovr",
        average="macro"
    )
    pr_auc = average_precision_score(
        y_test_bin,
        y_score,
        average="macro"
    )

    print(f"Macro ROC-AUC (OvR): {roc_auc:.4f}")
    print(f"Macro PR-AUC: {pr_auc:.4f}")

    # ROC curves per class
    plt.figure(figsize=(10, 8))

    all_fpr = np.unique(
        np.concatenate([
            roc_curve(y_test_bin[:, class_index], y_score[:, class_index])[0]
            for class_index in range(n_classes)
        ])
    )

    mean_tpr = np.zeros_like(all_fpr)

    for class_index, class_name in enumerate(malware_names):
        fpr, tpr, _ = roc_curve(
            y_test_bin[:, class_index],
            y_score[:, class_index]
        )
        class_roc_auc = auc(fpr, tpr)
        mean_tpr += np.interp(all_fpr, fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{class_name} (AUC = {class_roc_auc:.3f})"
        )

    mean_tpr /= n_classes
    macro_curve_auc = auc(all_fpr, mean_tpr)

    plt.plot(
        all_fpr,
        mean_tpr,
        linestyle="--",
        linewidth=2,
        label=f"Macro-average (AUC = {macro_curve_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], linestyle=":", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curves (One-vs-Rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve_LDA+NB1000.png")
    print("ROC curve saved as 'roc_curve_LDA+NB1000.png'")
    plt.close()

    # Precision-Recall curves per class
    plt.figure(figsize=(10, 8))

    for class_index, class_name in enumerate(malware_names):
        precision, recall, _ = precision_recall_curve(
            y_test_bin[:, class_index],
            y_score[:, class_index]
        )
        class_pr_auc = average_precision_score(
            y_test_bin[:, class_index],
            y_score[:, class_index]
        )

        plt.plot(
            recall,
            precision,
            label=f"{class_name} (AP = {class_pr_auc:.3f})"
        )

    macro_baseline = np.mean(y_test_bin)

    plt.axhline(
        y=macro_baseline,
        linestyle="--",
        label=f"Macro baseline = {macro_baseline:.3f}"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multiclass Precision-Recall Curves (One-vs-Rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pr_curve_LDA+NB1000.png")
    print("Precision-Recall curve saved as 'pr_curve_LDA+NB1000.png'")
    plt.close()

# Save global AUC metrics
auc_results = pd.DataFrame({
    "metric": [
        "ROC-AUC" if n_classes == 2 else "Macro ROC-AUC (OvR)",
        "PR-AUC" if n_classes == 2 else "Macro PR-AUC"
    ],
    "value": [roc_auc, pr_auc]
})

auc_results.to_csv("auc_results_LDA+NB1000.csv", index=False)
print("AUC results saved in 'auc_results_LDA+NB1000.csv'")


# Evaluation
print("Accuracy LDA + NB:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=malware_names))

# Guardar LDA y modelo
joblib.dump(scaler, "modelo_scaler.pkl")
joblib.dump(lda, "modelo_LDA3.pkl")
joblib.dump(nb_model, "modelo_NB.pkl")
print("Modelos LDA y NB guardados.")

def plot_confusion_matrix(y_test, y_pred, malware_names, output_file="confusion_matrix_LDA+NB_custom.png"):
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=malware_names, yticklabels=malware_names, annot_kws={"size": 20})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Predictions")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix", fontsize=16)

    # Save image
    plt.savefig(output_file)
    print(f"Confusion matrix saved as '{output_file}'")
    plt.close()

# Visualizing the simulated accuracy
plt.figure(figsize=(14, 6))

# Calculate and display metrics by class
def calculate_metrics(y_test, y_pred, malware_names, output_file="metrics_results_LDA+NB1000.csv", image_output="metrics_per_class_LDA+NB1000.png"):
    """
    Calculates classification metrics by class and saves the results in files.
    """
    # Generate report
    report = classification_report(y_test, y_pred, output_dict=True, target_names=malware_names)
    results_df = pd.DataFrame(report).transpose()

    # Save results
    results_df.to_csv(output_file)
    print(f"Results saved in '{output_file}'")

    # Display metrics
    results_df[['precision', 'recall', 'f1-score']].iloc[:-1].plot(kind='bar', figsize=(12, 8))
    plt.title('Classification Metrics per Class', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('Value')
    plt.xlabel('Classes')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(image_output)
    print(f"Graph saved as '{image_output}'")
    plt.close()

# Calculate metrics
calculate_metrics(y_test, y_pred, malware_names)
plot_confusion_matrix(y_test, y_pred, malware_names)


# Deployment-oriented measurements. Inference includes scaler, LDA and classifier.
def predict_end_to_end(X_batch_raw):
    X_batch = scaler.transform(X_batch_raw)
    X_batch = lda.transform(X_batch)
    return nb_model.predict(X_batch)

inference_metrics = benchmark_inference(predict_end_to_end, X_test_raw)
artifact_files = ['modelo_scaler.pkl', 'modelo_LDA3.pkl', 'modelo_NB.pkl']
model_size_bytes = total_file_size_bytes(artifact_files)

deployment_metrics = base_run_metadata('LDA+NB', 'VM-packer', 42)
deployment_metrics.update({
    'n_train_samples': len(X_train_raw),
    'n_test_samples': len(X_test_raw),
    'n_input_features': X_train_raw.shape[1],
    'epochs_executed': None,
    'lda_components': X_train.shape[1],
    'preprocessing_time_s': preprocessing_measure['elapsed_s'],
    'model_fit_time_s': fitting_measure['elapsed_s'],
    'total_training_pipeline_time_s': preprocessing_measure['elapsed_s'] + fitting_measure['elapsed_s'],
    'training_time_per_epoch_s': None,
    'training_peak_rss_mb': max(preprocessing_measure['peak_rss_mb'], fitting_measure['peak_rss_mb']),
    'training_incremental_peak_rss_mb': max(preprocessing_measure['incremental_peak_rss_mb'], fitting_measure['incremental_peak_rss_mb']),
    'model_parameters': None,
    'model_size_bytes': model_size_bytes,
    'model_size_mb': model_size_bytes / (1024 ** 2),
})
deployment_metrics.update(inference_metrics)
save_metrics(deployment_metrics, csv_path='deployment_metrics_vm_packer.csv', json_path='deployment_metrics_LDA_NB_vm_packer.json')
