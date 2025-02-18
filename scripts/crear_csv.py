def procesar_archivo(archivo_entrada, archivo_salida, etiquetas):
    """
    Procesa un archivo de texto, agrupando cada 10 valores con una etiqueta.
    
    :param archivo_entrada: Ruta al archivo de entrada con un valor por fila.
    :param archivo_salida: Ruta al archivo de salida con los valores agrupados.
    :param etiquetas: Lista de etiquetas que se asignarán secuencialmente.
    """
    with open(archivo_entrada, 'r') as entrada, open(archivo_salida, 'w') as salida:
        valores = []
        etiqueta_index = 0
        
        for linea in entrada:
            valores.append(linea.strip())
            
            if len(valores) == 1000:
                etiqueta = etiquetas[etiqueta_index % len(etiquetas)]  # Etiquetas cíclicas
                salida.write(','.join(valores) + f',{etiqueta}\n')
                valores = []
                etiqueta_index += 1
        
        if valores:
            print("Advertencia: Quedaron valores sin agrupar al final del archivo.")

# Uso del script
archivo_entrada = './../capturas/family/WIRENET.TXT'  # Ruta al archivo de entrada
archivo_salida = './family_buenos/WIRENET.txt'  # Ruta al archivo de salida
etiquetas = ['wirenet']  # Lista de etiquetas

procesar_archivo(archivo_entrada, archivo_salida, etiquetas)