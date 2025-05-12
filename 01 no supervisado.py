# %%
# Importamos las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib as mpl
from sklearn.datasets import fetch_openml

# Configuramos el estilo de los gráficos
plt.style.use('seaborn-v0_8-whitegrid')

# %%
# ----------------------------------------------------------------
# PARTE 1: GENERADOR DE DATASET ALEATORIO MODIFICADO
# ----------------------------------------------------------------

def generate_custom_dataset(n_centers=8, n_samples=2000, center_box=(-8, 8), random_state=42):

    # Aseguramos que el número de centroides esté entre 1 y 20
    n_centers = max(1, min(20, n_centers))

    # Generamos centroides con una distancia mínima entre ellos
    np.random.seed(random_state)
    min_distance = 3.0  # Distancia mínima entre centroides

    # Generamos los centroides asegurando la distancia mínima
    centers = []
    max_attempts = 1000  # Número máximo de intentos para posicionar cada centroide

    for _ in range(n_centers):
        attempts = 0
        while attempts < max_attempts:
            # Generamos un candidato para el nuevo centroide
            candidate = np.random.uniform(center_box[0], center_box[1], size=(2,))

            # Si es el primer centroide, lo aceptamos
            if not centers:
                centers.append(candidate)
                break

            # Calculamos la distancia mínima a los centroides existentes
            distances = [np.linalg.norm(candidate - center) for center in centers]
            min_dist = min(distances)

            # Si la distancia es suficiente, aceptamos el candidato
            if min_dist >= min_distance:
                centers.append(candidate)
                break

            attempts += 1

        # Si no pudimos colocar este centroide tras muchos intentos, reducimos la restricción
        if attempts == max_attempts:
            min_distance *= 0.9
            centers.append(np.random.uniform(center_box[0], center_box[1], size=(2,)))

    # Convertimos a array de numpy
    centers = np.array(centers)

    # Creamos desviaciones estándar aleatorias pero controladas para cada cluster
    cluster_std = np.random.uniform(0.1, 0.6, size=n_centers)

    # Generamos los datos usando make_blobs
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )

    return X, y, centers

# %%
# ----------------------------------------------------------------
# PARTE 2: FUNCIONES PARA VISUALIZACIÓN
# ----------------------------------------------------------------

def plot_clusters(X, y=None):

    plt.scatter(X[:, 0], X[:, 1], c=y, s=10, cmap='viridis', alpha=0.7)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
    plt.title("Dataset Generado con Centroides Bien Separados", fontsize=14)

def plot_data(X):

    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2, alpha=0.5)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):

    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=3,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                           show_xlabels=True, show_ylabels=True):

    # Calculamos los límites del gráfico
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1

    # Creamos una cuadrícula
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))

    # Predecimos el cluster para cada punto de la cuadrícula
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Dibujamos las fronteras de decisión
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.contour(xx, yy, Z, linewidths=1, colors='k', alpha=0.3)

    # Dibujamos los datos
    plot_data(X)

    # Dibujamos los centroides si se solicita
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    # Configuramos las etiquetas
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)

    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

def plot_clusterer_comparison(clusterer1, clusterer2, X, title1=None, title2=None):

    # Entrenamos los modelos
    clusterer1.fit(X)
    clusterer2.fit(X)

    # Creamos una figura con dos subplots
    plt.figure(figsize=(12, 5))

    # Primer subplot
    plt.subplot(121)
    plot_decision_boundaries(clusterer1, X)
    if title1:
        plt.title(title1, fontsize=14)

    # Segundo subplot
    plt.subplot(122)
    plot_decision_boundaries(clusterer2, X, show_ylabels=False)
    if title2:
        plt.title(title2, fontsize=14)

# %%
# ----------------------------------------------------------------
# PARTE 3: GENERAMOS Y VISUALIZAMOS EL DATASET
# ----------------------------------------------------------------

# Generamos el dataset modificado con centroides bien separados
n_centers = 8  # Puede ser entre 1 y 20
X, y_true, centers = generate_custom_dataset(n_centers=n_centers, n_samples=2000)

# Visualizamos el dataset generado
plt.figure(figsize=(10, 6))
plot_clusters(X, y_true)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.8, marker='+')
plt.title(f"Dataset con {n_centers} Centroides Bien Separados", fontsize=16)
plt.show()


# %%
# ----------------------------------------------------------------
# PARTE 4: APLICAMOS K-MEANS CON EL NÚMERO CORRECTO DE CLUSTERS
# ----------------------------------------------------------------

# Creamos y entrenamos el modelo K-Means
kmeans = KMeans(n_clusters=n_centers, random_state=42)
y_pred = kmeans.fit_predict(X)

# Visualizamos los resultados
plt.figure(figsize=(10, 6))
plot_decision_boundaries(kmeans, X)
plt.title(f"K-Means con {n_centers} Clusters", fontsize=16)
plt.show()


# %%
# ----------------------------------------------------------------
# PARTE 5: APLICAMOS SOFT CLUSTERING (TRANSFORMACIÓN DE DISTANCIAS)
# ----------------------------------------------------------------

# Generamos algunos nuevos puntos para demostrar el soft clustering
X_new = np.array([
    [0, 0],  # punto central
    [3, 3],  # esquina superior derecha
    [-3, 3],  # esquina superior izquierda
    [-3, -3],  # esquina inferior izquierda
    [3, -3]   # esquina inferior derecha
])

# Mostramos las predicciones para estos nuevos puntos
print("Predicciones para nuevos puntos:")
print(kmeans.predict(X_new))

# Mostramos las distancias a todos los centroides (soft clustering)
distances = kmeans.transform(X_new)
print("\nDistancias a todos los centroides:")
print(distances)

# %%
# ----------------------------------------------------------------
# PARTE 6: VISUALIZAMOS EL PROCESO ITERATIVO DE K-MEANS
# ----------------------------------------------------------------

# Creamos modelos con diferentes números de iteraciones
kmeans_iter1 = KMeans(n_clusters=n_centers, init="k-means++", n_init=1,
                      algorithm="elkan", max_iter=1, random_state=42)
kmeans_iter2 = KMeans(n_clusters=n_centers, init="k-means++", n_init=1,
                      algorithm="elkan", max_iter=2, random_state=42)
kmeans_iter3 = KMeans(n_clusters=n_centers, init="k-means++", n_init=1,
                      algorithm="elkan", max_iter=3, random_state=42)

# Entrenamos los modelos
kmeans_iter1.fit(X)
kmeans_iter2.fit(X)
kmeans_iter3.fit(X)

# Visualizamos las iteraciones
plt.figure(figsize=(15, 10))

# Primera iteración - centroides iniciales
plt.subplot(321)
plot_data(X)
plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.tick_params(labelbottom=False)
plt.title("Inicialización de Centroides (k-means++)", fontsize=14)

# Primera iteración - asignación de instancias
plt.subplot(322)
plot_decision_boundaries(kmeans_iter1, X, show_xlabels=False, show_ylabels=False)
plt.title("Asignación de Instancias (Iteración 1)", fontsize=14)

# Segunda iteración - actualización de centroides
plt.subplot(323)
plot_decision_boundaries(kmeans_iter1, X, show_centroids=False, show_xlabels=False)
plot_centroids(kmeans_iter2.cluster_centers_)
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.title("Actualización de Centroides (Iteración 2)", fontsize=14)

# Segunda iteración - asignación de instancias
plt.subplot(324)
plot_decision_boundaries(kmeans_iter2, X, show_xlabels=False, show_ylabels=False)
plt.title("Asignación de Instancias (Iteración 2)", fontsize=14)

# Tercera iteración - actualización de centroides
plt.subplot(325)
plot_decision_boundaries(kmeans_iter2, X, show_centroids=False)
plot_centroids(kmeans_iter3.cluster_centers_)
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.title("Actualización de Centroides (Iteración 3)", fontsize=14)

# Tercera iteración - asignación de instancias
plt.subplot(326)
plot_decision_boundaries(kmeans_iter3, X, show_ylabels=False)
plt.xlabel("$x_1$", fontsize=14)
plt.title("Asignación de Instancias (Iteración 3)", fontsize=14)

plt.tight_layout()
plt.show()

# %%
# ----------------------------------------------------------------
# PARTE 7: DEMOSTRAMOS DEPENDENCIA DE LA INICIALIZACIÓN
# ----------------------------------------------------------------

# Creamos modelos con diferentes inicializaciones aleatorias
kmeans_rnd_init1 = KMeans(n_clusters=n_centers, init="random", n_init=1,
                          algorithm="elkan", random_state=11)
kmeans_rnd_init2 = KMeans(n_clusters=n_centers, init="random", n_init=1,
                          algorithm="elkan", random_state=19)

# Comparamos las soluciones
plot_clusterer_comparison(kmeans_rnd_init1, kmeans_rnd_init2, X,
                         "Solución 1 (inicialización random_state=11)",
                         "Solución 2 (inicialización random_state=19)")
plt.show()

# %%
# ----------------------------------------------------------------
# PARTE 8: DEMOSTRAMOS MULTI-INICIALIZACIÓN
# ----------------------------------------------------------------

# Modelo con múltiples inicializaciones aleatorias
kmeans_rnd_10_inits = KMeans(n_clusters=n_centers, init="random", n_init=10,
                            algorithm="elkan", random_state=42)
kmeans_rnd_10_inits.fit(X)

# Visualizamos la mejor solución
plt.figure(figsize=(10, 6))
plot_decision_boundaries(kmeans_rnd_10_inits, X)
plt.title("K-Means con 10 Inicializaciones Aleatorias", fontsize=16)
plt.show()

# %%
# ----------------------------------------------------------------
# PARTE 9: MÉTODO DEL CODO PARA ENCONTRAR EL NÚMERO ÓPTIMO DE CLUSTERS
# ----------------------------------------------------------------
# Método del codo con flecha señalando el valor óptimo de k
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bo-', label='Inercia')

# Supongamos que, por análisis visual, elegimos k = 8 como el valor óptimo
optimal_k = 8
inertia_optimal = inertias[optimal_k - 1]

# Marcar el punto óptimo
plt.plot(optimal_k, inertia_optimal, 'ro', markersize=10, label=f'Punto de codo (k={optimal_k})')

# Añadir flecha y texto para resaltar el codo
plt.annotate('Posible valor óptimo de k',
             xy=(optimal_k, inertia_optimal),
             xytext=(optimal_k + 2, inertia_optimal + 500),
             arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=8),
             fontsize=12,
             horizontalalignment='left',
             verticalalignment='bottom')

# Configuración del gráfico
plt.xlabel("Número de clusters (k)", fontsize=14)
plt.ylabel("Inercia", fontsize=14)
plt.title("Método del Codo con Indicación del Valor Óptimo de k", fontsize=16)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# ----------------------------------------------------------------
# PARTE 10: SILHOUETTE SCORE MEJORADO CON INDICACIÓN DEL K ÓPTIMO
# ----------------------------------------------------------------

# Entrenamos modelos para diferentes valores de k
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X) for k in range(1, 21)]

# Calculamos el silhouette score para cada k (excepto k=1)
silhouette_scores = [silhouette_score(X, model.labels_) for model in kmeans_per_k[1:]]

# Encontramos el k con mejor silhouette score
optimal_k_silhouette = np.argmax(silhouette_scores) + 2  # +2 porque empezamos en k=2

# Visualizamos los scores con indicación del óptimo
plt.figure(figsize=(12, 6))
plt.plot(range(2, 21), silhouette_scores, "bo-")
plt.xlabel("Número de clusters (k)", fontsize=14)
plt.ylabel("Silhouette Score", fontsize=14)
plt.title("Silhouette Score para Diferentes Valores de k", fontsize=16)
plt.grid(True)

# Añadimos indicación del punto óptimo
plt.annotate(f'Mejor k = {optimal_k_silhouette}\nScore = {silhouette_scores[optimal_k_silhouette-2]:.3f}',
             xy=(optimal_k_silhouette, silhouette_scores[optimal_k_silhouette-2]),
             xytext=(optimal_k_silhouette+3, silhouette_scores[optimal_k_silhouette-2]-0.05),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=12, color='red')
plt.scatter([optimal_k_silhouette], [silhouette_scores[optimal_k_silhouette-2]], c='red', s=100)

plt.show()

# %%
# ----------------------------------------------------------------
# PARTE 11: DIAGRAMAS DE SILUETA COMPLETOS (PARA TODOS LOS K)
# ----------------------------------------------------------------

# Creamos una figura grande para todos los diagramas
plt.figure(figsize=(20, 25))
plt.suptitle("Diagramas de Silueta para k = 2 a 20", fontsize=20, y=1.02)

# Generamos un subplot para cada valor de k
for k in range(2, 21):
    plt.subplot(5, 4, k-1)

    # Obtenemos las etiquetas de cluster
    y_pred = kmeans_per_k[k-1].labels_

    # Calculamos los coeficientes de silueta
    silhouette_coefficients = silhouette_samples(X, y_pred)

    # Parámetros para la visualización
    padding = len(X) // 30
    pos = padding
    ticks = []

    # Para cada cluster, dibujamos los coeficientes de silueta
    for j in range(k):
        coeffs = silhouette_coefficients[y_pred == j]
        coeffs.sort()

        color = mpl.cm.Spectral(j / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    # Configuramos los ticks del eje y
    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))

    # Añadimos etiqueta cada 4 gráficos
    if (k-1) % 4 == 0:
        plt.ylabel("Cluster", fontsize=12)

    # Línea vertical para el score promedio
    plt.axvline(x=silhouette_scores[k-2], color="red", linestyle="--", linewidth=1)

    # Marcamos el mejor k con un borde especial
    if k == optimal_k_silhouette:
        for spine in plt.gca().spines.values():
            spine.set_color('red')
            spine.set_linewidth(3)

    plt.title(f"k = {k} (Score = {silhouette_scores[k-2]:.3f})", fontsize=12)
    plt.xlim(-0.1, 1)
    plt.ylim(0, pos)

    # Solo mostramos etiquetas en los gráficos inferiores
    if k > 16:
        plt.xlabel("Coeficiente de Silueta", fontsize=10)
    else:
        plt.tick_params(labelbottom=False)

plt.tight_layout()
plt.show()

# Mostramos los resultados óptimos
print("\nRESULTADOS ÓPTIMOS:")
print(f"- Según el método del codo: k = {optimal_k_elbow}")
print(f"- Según silhouette score: k = {optimal_k_silhouette}")
print(f"- Número real de clusters: {n_centers}")


