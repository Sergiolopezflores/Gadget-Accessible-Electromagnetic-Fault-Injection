from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# Names of malware types in order of classes
malware_names = ["ddos", "goodware", "ransomware", "rootkits", "spyware"]
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

# Load and prepare data
X, y_one_hot = load_data('one-hot-encoding1000.txt')
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert one-hot labels to indices
y = np.argmax(y_one_hot, axis=1)

# Dimensionality reduction with LDA
n_classes = len(np.unique(y))  # Number of classes
n_components = min(X.shape[1], n_classes - 1)  # Ensure valid limits for LDA
lda = LDA(n_components=n_components)
X_lda = lda.fit_transform(X, y)

# Classification with Random Forest
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Evaluation
print("Accuracy LDA + Random Forest:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=malware_names))

def plot_confusion_matrix(y_test, y_pred, malware_names, output_file="confusion_matrix_LDA+RF.png"):
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
def calculate_metrics(y_test, y_pred, malware_names, output_file="metrics_results_LDA+RandomForest.csv", image_output="metrics_per_class_LDA+RandomForest.png"):
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

