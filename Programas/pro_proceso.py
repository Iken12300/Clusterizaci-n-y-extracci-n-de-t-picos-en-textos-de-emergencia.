"""
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Trabajo de Luis Guerrero y Fabian Muñoz
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Librerías
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans

import seaborn as sns
import matplotlib.pyplot as plt
"""
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Procesos para Encontrar las Palabras Clave (p2)
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
def p2_palabras_clave(llamada, modelo, num_palabs):
    palabras = llamada.split()
    
    emb_palabras = modelo.encode(palabras)
    emb_llamada = modelo.encode(llamada)
    
    similaridades = util.pytorch_cos_sim(emb_llamada, emb_palabras)[0]
    top_indices = np.argsort(-similaridades)[:num_palabs]

    palabs_clave = [palabras[idx] for idx in top_indices]
    
    return palabs_clave
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def p2_recolectar_palabras_clave(dataset, modelo, num_palabs):
    ls_palabs_clave = []
    ls_palabs_tot = []
    for fila in dataset['ATA_TEXTO']:
        palabras_clave = p2_palabras_clave(fila, modelo, num_palabs)
        ls_palabs_clave.append(palabras_clave)
        ls_palabs_tot.extend(palabras_clave)
    return ls_palabs_clave, ls_palabs_tot
"""
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Creación de Clúster y Coeficiente de Silueta (p3)
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
def p3_crear_cluster(emb_lista, lista_clave):
    num_clsuters = int(input("Escriba el número de clusters que desea: "))

    kmeans = KMeans(n_clusters=num_clsuters)
    kmeans.fit(emb_lista)

    centroides = kmeans.cluster_centers_
    
    # Calculamos la similitud del coseno entre los embeddings y cada centroide
    similarity_to_centroids = cosine_similarity(emb_lista, centroides)

    counts = np.bincount(kmeans.labels_)
    for i, centroide in enumerate(centroides):
        most_similar_word_idx = np.argmax(similarity_to_centroids[:, i])
        most_similar_word = lista_clave[most_similar_word_idx]
        print(f"La palabra más cercana al centroide {i} es: {most_similar_word[0]} con {counts[i]} datos en este clúster.")

    silhouette_vals = p3_coeficiente_silueta(emb_lista, kmeans)

    oav_centroide = []
    oav_palabra = []
    oav_distancia = []

    for i in range(num_clsuters):

        most_similar_word_idx = np.argmax(similarity_to_centroids[:, i])
        most_similar_word = lista_clave[most_similar_word_idx][0]

        for idx, label in enumerate(kmeans.labels_):
            if label == i:
                # Añade la información al centroide correspondiente
                oav_centroide.append(most_similar_word)
                
                # Añade la palabra más cercana al centroide
                oav_palabra.append(lista_clave[idx][0])

                # Redimensionar
                centroide_i = centroides[i].reshape(1, -1)
                punto_datos = emb_lista[idx].reshape(1, -1)

                distancia = cosine_distances(punto_datos, centroide_i)
                oav_distancia.append(round(distancia[0][0], 4))

    data_to_append = []

    for i in range(len(oav_centroide)):
        # Añadir un diccionario con los datos al listado
        data_to_append.append({"CENTROIDE": oav_centroide[i], "PALABRA": oav_palabra[i], "DISTANCIA": oav_distancia[i]})

    # Crear el DataFrame fuera del bucle for con todos los datos
    OAV = pd.DataFrame(data_to_append, columns=["CENTROIDE", "PALABRA", "DISTANCIA"])
    OAV.to_csv("C:/Users/Luis/Desktop/gestco_examen/3_programa/oav.csv", index=False)

    p4_grafica_clusters(emb_lista, kmeans, centroides, similarity_to_centroids, lista_clave)
    p4_grafica_silueta(kmeans, silhouette_vals)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def p3_coeficiente_silueta(emb_lista, kmeans):
    # Calculamos los coeficientes de silueta para cada punto
    silhouette_vals = silhouette_samples(emb_lista, kmeans.labels_)

    # Creamos un diccionario para almacenar los coeficientes de silueta por clúster
    silhouette_by_cluster = {}
    for label, silhouette_val in zip(kmeans.labels_, silhouette_vals):
        silhouette_by_cluster.setdefault(label, []).append(silhouette_val)

    # Calculamos el coeficiente de silueta promedio para cada clúster
    for cluster_id in silhouette_by_cluster:
        avg_silhouette = sum(silhouette_by_cluster[cluster_id]) / len(silhouette_by_cluster[cluster_id])
        print(f"Clúster {cluster_id}: Coeficiente de Silueta promedio = {avg_silhouette}")
    
    return silhouette_vals
"""
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Gráficos (p4):
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
def p4_grafica_clusters(emb_lista, kmeans, centroides, similarity_to_centroids, lista_clave):
    emb_lista = np.array(emb_lista)
    plt.scatter(emb_lista[:, 0], emb_lista[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
    plt.scatter(centroides[:, 0], centroides[:, 1], c='red', marker='x')
    for i, centroide in enumerate(centroides):
        plt.gca().add_artist(plt.Circle((centroide[0], centroide[1]), 0.5, color='r', fill=False))
        most_similar_word_idx = np.argmax(similarity_to_centroids[:, i])
        most_similar_word = lista_clave[most_similar_word_idx]
        plt.text(centroide[0], centroide[1], most_similar_word[0], fontsize=12)

    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.title('Clústers con Centroides')
    plt.show()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def p4_grafica_silueta(kmeans, silhouette_vals):
    fig, ax1 = plt.subplots(1, 1)
    y_lower = 10

    for i in range(kmeans.n_clusters):
        ith_cluster_silhouette_values = silhouette_vals[kmeans.labels_ == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.nipy_spectral(float(i) / kmeans.n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title("Gráfica del Coeficiente de Silueta")
    ax1.set_xlabel("Coeficiente de Silueta")
    ax1.set_ylabel("Clúster")
    ax1.axvline(x=silhouette_vals.mean(), color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8])
    plt.show()
"""
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Main
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
if __name__ == "__main__":
    # Modelos a ocupar
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    #model = SentenceTransformer('jinaai/jina-embeddings-v2-base-es', trust_remote_code=True)
    #model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    #model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')

    # Lectura de los datos
    dataset = pd.read_csv(r"C:/Users/Luis/Desktop/gestco_examen/3_programa/conversaciones.csv", sep=";", on_bad_lines="warn")
    dataset = dataset[dataset['ATA_TEXTO'].str.split().str.len() >= 5]
    #dataset = dataset.head(100)

    # Recolectar todas las palabras clave en una super lista
    ls_palabs_clave, ls_palabs_tot = p2_recolectar_palabras_clave(dataset, model, 5)

    """
    Ejemplo con 1 registro y 3 palabras clave

    array([
            [ 0.03254424,  0.01141798, -0.00854072, ...,  0.0234774 , -0.0477055 ,  0.01877508],
            [ 0.03254424,  0.01141798, -0.00854072, ...,  0.0234774 , -0.0477055 ,  0.01877508],
            [ 0.03254424,  0.01141798, -0.00854072, ...,  0.0234774 , -0.0477055 ,  0.01877508]
                ], dtype=float32),
        
    array([ 0.03254424,  0.01141798, -0.00854072, ...,  0.0234774 , -0.0477055 ,  0.01877508], dtype=float32), 
    """

    emb_lista_clave = [model.encode(sublista) for sublista in ls_palabs_clave]
    # Aplanar cada sublista de embeddings en un solo vector
    emb_lista_clave = [np.concatenate(sublista) for sublista in emb_lista_clave]

    p3_crear_cluster(emb_lista_clave, ls_palabs_clave)
