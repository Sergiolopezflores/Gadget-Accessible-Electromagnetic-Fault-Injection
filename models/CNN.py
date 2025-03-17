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

# Nombres de los tipos de malware en el orden de las clases
nombres_malware = ["not_packed", "packed"] 

# Función para cargar datos
def cargar_datos(archivo):
    datos = []
    etiquetas = []
    contador_etiquetas = Counter()
    etiquetas_clases = {}  # Diccionario para mapear etiquetas one-hot a clases

    with open(archivo, 'r') as f:
        for i, linea in enumerate(f):
            elementos = linea.strip().split(',')
            vector = list(map(int, elementos[:1000]))
            etiqueta_one_hot = tuple(map(float, elementos[1000:]))  # Convertir a tupla para usar en el contador
            
            if etiqueta_one_hot not in etiquetas_clases:
                etiquetas_clases[etiqueta_one_hot] = f"Clase_{len(etiquetas_clases)}"  # Asignar nombre a la clase

            datos.append(vector)
            etiquetas.append(etiqueta_one_hot)
            contador_etiquetas[etiqueta_one_hot] += 1

    # Mostrar el total de muestras por etiqueta
    print("\nDistribución de etiquetas:")
    for etiqueta, cantidad in contador_etiquetas.items():
        print(f"{etiqueta}: {cantidad}")

    return np.array(datos), np.array(etiquetas)

# Construir el modelo CNN
def construir_cnn(input_shape, num_classes):
    model = Sequential([
        # Primera capa convolucional
        Conv1D(128, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Dropout(0.4),

        # Segunda capa convolucional
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.3),

        # Tercera capa convolucional
        Conv1D(32, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.3),

        # Aplanado y capas densas
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer='l2'),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dropout(0.3),

        # Capa de salida
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Cargar y preparar datos
X, y = cargar_datos('one-hot-encoding1000.txt')
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Redimensionar para CNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo CNN
cnn_model = construir_cnn((X.shape[1], 1), y_train.shape[1])
historial_mejorado=cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, batch_size=32)

# Evaluación
y_pred = np.argmax(cnn_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("Accuracy CNN:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=nombres_malware))


# Visualizar la pérdida del entrenamiento y validación
plt.figure(figsize=(14, 6))

# Pérdida
plt.subplot(1, 2, 1)
plt.plot(historial_mejorado.history['loss'], label='Pérdida de entrenamiento')
plt.plot(historial_mejorado.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida durante el entrenamiento y validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.savefig("perdida_CNN_casero1000.png")
print(f"Gráfica guardada como perdida_CNN_casero1000")

# Precisión
plt.subplot(1, 2, 2)
plt.plot(historial_mejorado.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(historial_mejorado.history['val_accuracy'], label='Precisión de validación')
plt.title('Precisión durante el entrenamiento y validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()
plt.savefig("precision_CNN_casero1000.png")
print(f"Gráfica guardada como precision_CNN_casero1000")
plt.close()

def calcular_métricas(X_test, y_test, modelo, nombres_malware, archivo_salida="resultados_métricas_CNN_casero1000.csv", imagen_salida="metricas_por_clase_CNN_casero1000.png"):
    # Predicciones y métricas por clase
    predicciones = modelo.predict(X_test)
    etiquetas_reales = np.argmax(y_test, axis=1)
    etiquetas_predichas = np.argmax(predicciones, axis=1)

    reporte = classification_report(etiquetas_reales, etiquetas_predichas, output_dict=True, target_names=nombres_malware)
    resultados_df = pd.DataFrame(reporte).transpose()

    # Guardar resultados
    resultados_df.to_csv(archivo_salida)
    print(f"Resultados guardados en '{archivo_salida}'")

    # Visualizar métricas
    resultados_df[['precision', 'recall', 'f1-score']].iloc[:-1].plot(kind='bar', figsize=(12, 8))
    plt.title('Métricas de Clasificación por Clase')
    plt.ylabel('Valor')
    plt.xlabel('Clases')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(imagen_salida)
    print(f"Gráfica guardada como '{imagen_salida}'")
    plt.close()

def graficar_matriz_confusion(y_test, y_pred, nombres_malware, archivo_salida="matriz_confusion_CNN_casero1000.png"):
    # Calcular matriz de confusión
    matriz_confusion = confusion_matrix(y_test, y_pred)

    # Graficar matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="Blues", xticklabels=nombres_malware, yticklabels=nombres_malware)
    plt.xlabel("Predicciones")
    plt.ylabel("Valores reales")
    plt.title("Matriz de Confusión")

    # Guardar imagen
    plt.savefig(archivo_salida)
    print(f"Matriz de confusión guardada como '{archivo_salida}'")
    plt.close()



# Calcular métricas
calcular_métricas(X_test, y_test, cnn_model, nombres_malware)
graficar_matriz_confusion(y_true, y_pred, nombres_malware)
