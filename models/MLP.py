import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter

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

# Building the MLP model
def build_mlp(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and prepare data
X, y = load_data('one-hot-encoding1000.txt')
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the MLP model
mlp_model = build_mlp(X_train.shape[1], y_train.shape[1])
best_history = mlp_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, batch_size=32)

# Evaluation
y_pred = np.argmax(mlp_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("MLP Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=malware_names))

# Visualizing training and validation loss
plt.figure(figsize=(14, 6))

# Loss
plt.subplot(1, 2, 1)
plt.plot(best_history.history['loss'], label='Training Loss')
plt.plot(best_history.history['val_loss'], label='Validation Loss')
plt.title('Loss during Training and Validation', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_MLP.png")
print("Graph saved as loss_MLP.png")

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(best_history.history['accuracy'], label='Training Accuracy')
plt.plot(best_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy during Training and Validation', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("accuracy_MLP.png")
print("Graph saved as accuracy_MLP.png")
plt.close()

def compute_metrics(X_test, y_test, model, malware_names, output_file="metrics_results_MLP.csv", image_output="metrics_per_class_MLP.png"):
    # Predictions and per-class metrics
    predictions = model.predict(X_test)
    true_labels = np.argmax(y_test, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)

    report = classification_report(true_labels, predicted_labels, output_dict=True, target_names=malware_names)
    results_df = pd.DataFrame(report).transpose()

    # Save results to CSV with five decimal places
    pd.options.display.float_format = '{:.5f}'.format
    results_df.to_csv(output_file, float_format='%.5f')
    print(f"Results saved in '{output_file}'")

    # Display per-class metrics
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

def plot_confusion_matrix(y_test, y_pred, malware_names, output_file="confusion_matrix_MLP.png"):
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=malware_names, yticklabels=malware_names, annot_kws={"size": 20})
    plt.xticks(fontsize=14)  # Aumenta el tamaño de los nombres en el eje X
    plt.yticks(fontsize=14)  # Aumenta el tamaño de los nombres en el eje Y
    plt.xlabel("Predictions")
    plt.ylabel("True Labels")
    plt.xticks(rotation=35, ha="right")
    plt.title("Confusion Matrix", fontsize=16)

    # Save image
    plt.savefig(output_file)
    print(f"Confusion matrix saved as '{output_file}'")
    plt.close()

# Compute metrics
compute_metrics(X_test, y_test, mlp_model, malware_names)
plot_confusion_matrix(y_true, y_pred, malware_names)

