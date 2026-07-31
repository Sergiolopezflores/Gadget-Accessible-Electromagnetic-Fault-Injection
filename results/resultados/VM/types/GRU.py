from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter

# Names of malware types in order of classes
malware_names = ["ddos", "goodware", "ransomware", "rootkits", "spyware"]

# Function to load and normalize data
def load_data(file):
    data = []
    labels = []
    label_counter = Counter()
    label_classes = {}  # Dictionary for mapping one-hot tags to classes

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

# Building the GRU model
def build_gru(input_shape, num_classes):
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.4),
        
        GRU(64, return_sequences=True),
        Dropout(0.3),

        GRU(32),
        Dropout(0.3),

        Dense(128, activation='relu'),
        Dropout(0.4),
        
        Dense(64, activation='relu'),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Load and prepare sequential data
X, y = load_data('one-hot-encoding1000.txt')
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for GRU (samples, timesteps, features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train GRU model
gru_model = build_gru((X.shape[1], 1), y_train.shape[1])
history_gru = gru_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, batch_size=32)

# Model evaluation
y_pred = np.argmax(gru_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("GRU Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=malware_names))

# Save model
gru_model.save("gru_model_custom.h5")

# Visualize loss and accuracy
plt.figure(figsize=(14, 6))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history_gru.history['loss'], label='Training Loss')
plt.plot(history_gru.history['val_loss'], label='Validation Loss')
plt.title('Loss during Training and Validation', fontsize=21)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.legend()
plt.savefig("loss_GRU_custom.png")

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history_gru.history['accuracy'], label='Training Accuracy')
plt.plot(history_gru.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy during Training and Validation', fontsize=21)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.legend()

plt.tight_layout()
plt.savefig("accuracy_GRU_custom.png")
plt.close()

# Function to compute and save metrics
def compute_metrics(X_test, y_test, model, malware_names, output_file="metrics_results_GRU_custom.csv", image_output="class_metrics_GRU_custom.png"):
    predictions = model.predict(X_test)
    true_labels = np.argmax(y_test, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)

    report = classification_report(true_labels, predicted_labels, output_dict=True, target_names=malware_names)
    results_df = pd.DataFrame(report).transpose()
    pd.options.display.float_format = '{:.5f}'.format

    results_df.to_csv(output_file, float_format='%.5f')
    print(f"Results saved in '{output_file}'")

    results_df[['precision', 'recall', 'f1-score']].iloc[:-1].plot(kind='bar', figsize=(12, 8))
    plt.title('Classification Metrics by Class', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('Value', fontsize=18)
    plt.xlabel('Classes', fontsize=18)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(image_output)
    print(f"Graph saved as '{image_output}'")
    plt.close()

# Function to plot and save confusion matrix
def plot_confusion_matrix(y_test, y_pred, malware_names, output_file="confusion_matrix_GRU_custom.png"):
    confusion_mtx = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues", xticklabels=malware_names, yticklabels=malware_names, annot_kws={"size": 20})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Predictions", fontsize=18)
    plt.ylabel("True Labels", fontsize=18)
    plt.title("Confusion Matrix", fontsize=16)
    
    plt.savefig(output_file)
    print(f"Confusion matrix saved as '{output_file}'")
    plt.close()

# Compute metrics
compute_metrics(X_test, y_test, gru_model, malware_names)
plot_confusion_matrix(y_true, y_pred, malware_names)

