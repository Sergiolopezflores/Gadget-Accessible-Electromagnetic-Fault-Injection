from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
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
        for line in f:
            elements = line.strip().split(',')
            vector = list(map(int, elements[:1000]))
            one_hot_label = tuple(map(float, elements[1000:])) # Convert to tuple for use in the counter
            
            if one_hot_label not in label_classes:
                label_classes[one_hot_label] = f"Class_{len(label_classes)}"  # Assign name to the class
                
            
            data.append(vector)
            labels.append(one_hot_label)
            label_counter[one_hot_label] += 1
            
            
    # Show the total number of samples per label
    print("\nLabel distribution:")
    for label, count in label_counter.items():
        print(f"{label}: {count}")
            
    X = np.array(data)
    y = np.array(labels)

    # Normalize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


# Building the RNN model
def build_improved_model(input_dim, num_classes):
    model = Sequential()

    # Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(128, return_sequences=False), input_shape=input_dim))
    model.add(Dropout(0.3))  # Add dropout to prevent overfitting

    # Dense layers with L2 regularization
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))  # Add additional Dropout

    # Output layer with X neurons (X classes)
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Load data from one-hot-encoding.txt file
X, y = load_data('one-hot-encoding1000.txt')

# Reshape X to be compatible with LSTM (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the bidirectional LSTM model
improved_model = build_improved_model((X.shape[1], 1), y_train.shape[1])

# Train the improved model with callbacks
best_history = improved_model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=32,
    validation_data=(X_test, y_test),
)

y_pred = np.argmax(improved_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("RNN Accuracy:", accuracy_score(y_true, y_pred))
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
plt.savefig("loss_RNN.png")
print("Graph saved as loss_RNN.png")

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
plt.savefig("accuracy_RNN.png")
print("Graph saved as accuracy_RNN.png")
plt.close()

def compute_metrics(X_test, y_test, model, malware_names, output_file="metrics_results_RNN.csv", image_output="class_metrics_RNN.png"):
    # Predictions and per-class metrics
    predictions = model.predict(X_test)
    true_labels = np.argmax(y_test, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)

    report = classification_report(true_labels, predicted_labels, output_dict=True, target_names=malware_names)
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

def plot_confusion_matrix(y_test, y_pred, malware_names, output_file="confusion_matrix_RNN.png"):
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=malware_names, yticklabels=malware_names, annot_kws={"size": 20})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Predictions")
    plt.ylabel("True Labels")
    plt.xticks(rotation=35, ha="right")
    plt.title("Confusion Matrix", fontsize=16)

    # Save image
    plt.savefig(output_file)
    print(f"Confusion matrix saved as '{output_file}'")
    plt.close()

# Compute metrics
compute_metrics(X_test, y_test, improved_model, malware_names)
plot_confusion_matrix(y_true, y_pred, malware_names)

