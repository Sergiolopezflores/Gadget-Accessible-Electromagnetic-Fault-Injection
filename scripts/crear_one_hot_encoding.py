import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Función para cargar los datos desde el archivo de entrada
def cargar_datos(archivo_entrada):
    datos = []
    etiquetas = []

    # Abrir el archivo de entrada en modo lectura
    try:
        with open(archivo_entrada, 'r') as f_in:
            for linea in f_in:
                # Separar los números enteros de la etiqueta
                elementos = linea.strip().split(',')
                vector = list(map(int, elementos[:1000]))  # Los 1000 primeros son el vector de entrada
                etiqueta = elementos[1000]  # El último elemento es la etiqueta
                datos.append(vector)
                etiquetas.append(etiqueta)
    except Exception as e:
        print(f"Error al cargar los datos: {e}")

    return np.array(datos), np.array(etiquetas)

# Función para convertir las etiquetas a One-hot encoding
def convertir_etiquetas_a_one_hot(etiquetas):
    # Convertir las etiquetas a números
    label_encoder = LabelEncoder()
    etiquetas_numericas = label_encoder.fit_transform(etiquetas)

    # Convertir los números a one-hot encoding
    etiquetas_one_hot = to_categorical(etiquetas_numericas)

    return etiquetas_one_hot, label_encoder

# Función para escribir los datos y el one-hot encoding en un archivo de salida
def escribir_salida(datos, etiquetas_one_hot, archivo_salida):
    try:
        with open(archivo_salida, 'w') as f_out:
            for i in range(len(datos)):
                # Convertir el vector a una cadena de números separados por comas
                vector_str = ','.join(map(str, datos[i]))
                # Convertir el one-hot encoding a cadena
                etiqueta_one_hot_str = ','.join(map(str, etiquetas_one_hot[i]))
                # Escribir la línea al archivo
                f_out.write(f"{vector_str},{etiqueta_one_hot_str}\n")
    except Exception as e:
        print(f"Error al escribir los datos en el archivo: {e}")

# Función principal
def procesar_archivo(archivo_entrada, archivo_salida):
    # Cargar los datos y etiquetas
    datos, etiquetas = cargar_datos(archivo_entrada)

    # Convertir las etiquetas a one-hot encoding
    etiquetas_one_hot, _ = convertir_etiquetas_a_one_hot(etiquetas)

    # Escribir los datos y el one-hot encoding en el archivo de salida
    escribir_salida(datos, etiquetas_one_hot, archivo_salida)

# Ejecución del script
if __name__ == '__main__':
    archivo_entrada = './CSV/family_buenos/csv_final.txt'  # Archivo de entrada con los vectores y etiquetas
    archivo_salida = './CSV/family_buenos/one-hot-encoding1000.txt'  # Archivo de salida donde se escribirán los datos

    # Procesar el archivo de entrada y generar el archivo de salida
    procesar_archivo(archivo_entrada, archivo_salida)
