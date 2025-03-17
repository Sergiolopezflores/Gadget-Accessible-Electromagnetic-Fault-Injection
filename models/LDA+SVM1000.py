from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# Nombres de los tipos de malware en el orden de las clases
nombres_malware = ["ddos", "goodware", "ransmoware", "rootkits", "spyware"] 

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

# Cargar y preparar datos
X, y_one_hot = cargar_datos('one-hot-encoding1000.txt')
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convertir etiquetas one-hot a índices
y = np.argmax(y_one_hot, axis=1)

# Reducir dimensionalidad con LDA
n_classes = len(np.unique(y))  # Número de clases
n_components = min(X.shape[1], n_classes - 1)  # Asegurar límites válidos para LDA
lda = LDA(n_components=n_components)
X_lda = lda.fit_transform(X, y)

# Clasificación con SVM
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Evaluación
print("Accuracy LDA + SVM:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=nombres_malware))

def graficar_matriz_confusion(y_test, y_pred, nombres_malware, archivo_salida="matriz_confusion_LDA+SVM_casero.png"):
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

# Visualizar la precisión simulada
plt.figure(figsize=(14, 6))

# Calcular y visualizar métricas por clase
def calcular_métricas(y_test, y_pred, nombres_malware, archivo_salida="resultados_métricas_LDA+SVM1000.csv", imagen_salida="metricas_por_clase_LDA+SVM1000.png"):
    """
    Calcula métricas de clasificación por clase y guarda los resultados en archivos.
    """
    # Generar reporte
    reporte = classification_report(y_test, y_pred, output_dict=True, target_names=nombres_malware)
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

# Calcular métricas
calcular_métricas(y_test, y_pred, nombres_malware)
graficar_matriz_confusion(y_test, y_pred, nombres_malware)
