from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.optimizers import Adam
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

# Building the CNN model
def build_cnn(input_shape, num_classes):
    model = Sequential([
        # First convolutional layer
        Conv1D(128, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Dropout(0.4),

        # Second convolutional layer
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.3),

        # Third convolutional layer
        Conv1D(32, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.3),

        # Flattening and dense layers
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer='l2'),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dropout(0.3),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and prepare data
X, y = load_data('one-hot-encoding1000.txt')
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for CNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the CNN model
cnn_model = build_cnn((X.shape[1], 1), y_train.shape[1])
history = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, batch_size=32)

# Evaluation
y_pred = np.argmax(cnn_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("CNN Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=malware_names))

# Visualize training and validation loss
plt.figure(figsize=(14, 6))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss during Training and Validation', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_CNN.png")
print("Graph saved as loss_CNN.png")

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy during Training and Validation', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("accuracy_CNN.png")
print("Graph saved as accuracy_CNN.png")
plt.close()

# Function to calculate classification metrics
def compute_metrics(X_test, y_test, model, malware_names, output_file="metrics_results_CNN.csv", output_image="metrics_by_class_CNN.png"):
    predictions = model.predict(X_test)
    real_labels = np.argmax(y_test, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)

    report = classification_report(real_labels, predicted_labels, output_dict=True, target_names=malware_names)
    results_df = pd.DataFrame(report).transpose()

    # Save results
    results_df.to_csv(output_file)
    print(f"Results saved in '{output_file}'")

    # Display metrics
    results_df[['precision', 'recall', 'f1-score']].iloc[:-1].plot(kind='bar', figsize=(12, 8))
    plt.title('Classification Metrics by Class', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('Value')
    plt.xlabel('Classes')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Graph saved as '{output_image}'")
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, malware_names, output_file="confusion_matrix_CNN.png"):
    confusion_mat = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=malware_names, yticklabels=malware_names, annot_kws={"size": 20})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Predictions")
    plt.ylabel("Actual Values")
    plt.xticks(rotation=35, ha="right")
    plt.title("Confusion Matrix", fontsize=16)

    plt.savefig(output_file)
    print(f"Confusion matrix saved as '{output_file}'")
    plt.close()

# Compute and save metrics
compute_metrics(X_test, y_test, cnn_model, malware_names)
plot_confusion_matrix(y_true, y_pred, malware_names)

