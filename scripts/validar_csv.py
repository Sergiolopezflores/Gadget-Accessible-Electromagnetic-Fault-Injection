import csv

# Función para validar el archivo CSV con vectores y one-hot encoding
def validar_formato_csv(archivo_entrada):
    try:
        with open(archivo_entrada, 'r') as f_in:
            reader = csv.reader(f_in)
            linea_numero = 1  # Mantener el registro de la línea para identificar errores
            longitud = 1005 #Cambiar dependiendo de las clases que tengamos (1000 números + X elementos one-hot)

            for linea in reader:
                # Comprobar que la longitud de cada línea sea XXXX (1000 números + X elementos one-hot)
                if len(linea) != longitud:
                    return f"Error en la línea {linea_numero}: La línea no tiene {longitud} elementos, tiene {len(linea)}"

                # Comprobar que los primeros 1000 elementos sean enteros
                for i in range(1000):
                    try:
                        int(linea[i])
                    except ValueError:
                        return f"Error en la línea {linea_numero}: El elemento {i+1} ('{linea[i]}') no es un número entero"

                # Comprobar que los últimos X elementos sean números flotantes (0.0 o 1.0)
                for j in range(1000, longitud):
                    try:
                        valor = float(linea[j])
                        if valor not in [0.0, 1.0]:
                            return f"Error en la línea {linea_numero}: El elemento {j+1} ('{linea[j]}') no es 0.0 o 1.0"
                    except ValueError:
                        return f"Error en la línea {linea_numero}: El elemento {j+1} ('{linea[j]}') no es un número válido"

                # Comprobar que solo hay un valor igual a 1.0 en el one-hot encoding
                if sum(float(linea[j]) for j in range(1000, longitud)) != 1.0:
                    return f"Error en la línea {linea_numero}: Los elementos de one-hot encoding no suman 1.0"

                linea_numero += 1

            # Si todas las comprobaciones pasan
            return "TODO OK"
    except Exception as e:
        return f"Error al abrir o leer el archivo: {e}"

# Archivo de entrada a validar
archivo_entrada = 'CSV/prueba1000/one-hot-encoding1000.txt'

# Validar el archivo y mostrar el resultado
resultado = validar_formato_csv(archivo_entrada)
print(resultado)
