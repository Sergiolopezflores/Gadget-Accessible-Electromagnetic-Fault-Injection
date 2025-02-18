import os

def agrupar_archivos(directorio_entrada, archivo_salida):
    """
    Agrupa todos los archivos .txt en un directorio en un solo archivo de salida.
    
    :param directorio_entrada: Ruta al directorio que contiene los archivos de texto.
    :param archivo_salida: Ruta del archivo de salida donde se guardarán los datos agrupados.
    """
    with open(archivo_salida, 'w') as salida:
        # Recorre todos los archivos en el directorio
        for archivo in os.listdir(directorio_entrada):
            if archivo.endswith('.txt'):  # Solo archivos .txt
                archivo_ruta = os.path.join(directorio_entrada, archivo)
                with open(archivo_ruta, 'r') as entrada:
                    # Escribe el contenido de cada archivo en el archivo de salida
                    salida.write(entrada.read())
    print(f"Todos los archivos de {directorio_entrada} han sido agrupados en {archivo_salida}")

# Uso del script
directorio_entrada = './family_buenos'  # Ruta al directorio de entrada con los archivos .txt
archivo_salida = 'csv_final.txt'  # Ruta al archivo de salida

agrupar_archivos(directorio_entrada, archivo_salida)
