# %% [markdown]
# ### K-Means para aprendizaje semi-supervisado.

# %% [markdown]
# El aprendizaje semi-supervisado (o *Semi-supervised Learning*) comprende el conjunto de técnicas que nos permiten entrenar modelos con datasets parcialmente etiquetados. En esta sección vamos a ver un ejemplo de como podemos aplicar esta técnica con el dataset MNIST y usando *K-Means*. Empezamos descargando el dataset.

# %% [markdown]
# librerias

# %% [markdown]
# # 20newsgroups

# %%
from sklearn.datasets import fetch_20newsgroups
import numpy as np

# Descargar el dataset (conjunto de entrenamiento)
newsgroups_train = fetch_20newsgroups(subset='train')
# Descargar el dataset (conjunto de prueba)
newsgroups_test = fetch_20newsgroups(subset='test')



# Mostrar las primeras etiquetas y ejemplos
print("Etiquetas: ", newsgroups_train.target_names)
print("Primeros documentos: ", newsgroups_train.data[:5])

# Separar los datos y etiquetas
X = newsgroups_train.data
y = newsgroups_train.target

X_test = newsgroups_test.data
y_test = newsgroups_test.target



# %%
# Fijar un porcentaje de datos etiquetados por clase
prop_etiquetado = 0.1  # 10% de los datos etiquetados
y_semi = np.full_like(y, fill_value=-1)  # Inicializar con -1 para no etiquetados

# Proporción de datos etiquetados
for clase in np.unique(y):
    idx = np.where(y == clase)[0]
    n = int(len(idx) * prop_etiquetado)
    seleccionados = np.random.choice(idx, n, replace=False)
    y_semi[seleccionados] = y[seleccionados]

# Ahora y_semi tendrá las etiquetas semi-supervisadas


# %%
print(newsgroups_train.data[0])  # Muestra el primer documento


# %%
import matplotlib.pyplot as plt
import numpy as np

# Contar la cantidad de documentos por clase
unique, counts = np.unique(newsgroups_train.target, return_counts=True)
class_distribution = dict(zip(newsgroups_train.target_names, counts))

# Graficar la distribución de clases
plt.figure(figsize=(10, 6))
plt.bar(class_distribution.keys(), class_distribution.values())
plt.title("Distribución de clases en 20 Newsgroups")
plt.xlabel("Clases")
plt.ylabel("Número de documentos")
plt.xticks(rotation=90)
plt.show()


# %%
from sklearn.manifold import TSNE

# Ajuste de t-SNE con diferentes parámetros
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
X_tsne = tsne.fit_transform(X.toarray())  # X es la matriz de características

# Graficar el resultado
plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=newsgroups_train.target, cmap='tab20', s=20)
plt.colorbar(label='Clases')
plt.title("Visualización t-SNE ajustada")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.show()



# %%
from sklearn.decomposition import PCA

# Reducir la dimensionalidad a 2D con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Graficar los puntos en 2D
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=newsgroups_train.target, cmap='tab20', s=20)
plt.colorbar(label='Clases')
plt.title("Visualización PCA de los documentos en 2D")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.show()


# %%
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Valores de k a probar
k_valores = range(1, 30)
inertia = []
silhouette_scores = []

# Iterar sobre diferentes valores de k
for k in k_valores:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)  # Ajustar el modelo a los datos
    inertia.append(kmeans.inertia_)  # Inercia
    if k > 1:
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))  # Silhouette Score
    else:
        silhouette_scores.append(0)  # Silhouette score no definido para k=1

    print(f"Iteración {k} completada.")

# %% [markdown]
# ### grafico de codo(inercia)

# %%
import matplotlib.pyplot as plt

# Graficar codo (inercia)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(k_valores, inertia, marker='o')
plt.xticks(k_valores)  # Muestra cada valor de k en el eje x
plt.title("Método del codo (Inercia)")
plt.xlabel("Número de clusters k")
plt.ylabel("Inercia")
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## grafico silhouette score

# %%
# Gráfico del Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(k_valores, silhouette_scores, marker='o', color='green')
plt.xticks(k_valores)
plt.title("Silhouette Score")
plt.xlabel("Número de clusters k")
plt.ylabel("Silhouette")
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# seleccionamos el mejor k entre los valores que se calcularon meidante el metodo del codo.

# %%
# Para obtener el valor de k óptimo según el codo y el silhouette score
# Método del codo: buscar donde la pendiente cambia significativamente
inertia_diff = [inertia[i] - inertia[i+1] for i in range(len(inertia)-1)]
optimal_k_codo = k_valores[3]  # Valor predeterminado en caso de no encontrar un codo claro
for i in range(1, len(inertia_diff)-1):
	if inertia_diff[i] < inertia_diff[i-1] * 0.5:  # Si la pendiente decrece significativamente
		optimal_k_codo = k_valores[i+1]
		break
optimal_k_silhouette = k_valores[silhouette_scores.index(max(silhouette_scores))]  # Mayor silhouette score

print(f"El valor óptimo de k según el método del codo es: {optimal_k_codo}")
print(f"El valor óptimo de k según el silhouette score es: {optimal_k_silhouette}")

# %%
from sklearn.cluster import KMeans

# Aplicar KMeans con ambos valores óptimos
kmeans_codo = KMeans(n_clusters=optimal_k_codo, random_state=42)
X_dist_codo = kmeans_codo.fit_transform(X)
labels_codo = kmeans_codo.labels_

kmeans_sil = KMeans(n_clusters=optimal_k_silhouette, random_state=42)
X_dist_sil = kmeans_sil.fit_transform(X)
labels_sil = kmeans_sil.labels_


# %%
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reducción a 2D para visualización
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X.toarray())  # Usamos X en lugar de X_train y lo convertimos a un array denso

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_codo, cmap='tab10', s=10)
plt.title(f"KMeans con k = {optimal_k_codo} (codo)")

plt.subplot(1, 2, 2)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_sil, cmap='tab10', s=10)
plt.title(f"KMeans con k = {optimal_k_silhouette} (silhouette)")

plt.tight_layout()
plt.show()


# %% [markdown]
# El siguiente paso consiste en anotar manualmente estas etiquetas (aquí haremos trampas ya que disponemos de dichas etiquetas :p).

# %%
# Suponiendo que ya aplicaste KMeans con el k óptimo
kmeans = KMeans(n_clusters=optimal_k_codo, random_state=42)
X_dist = kmeans.fit_transform(X)  # distancia de cada muestra a cada centroide
idxs = np.argmin(X_dist, axis=0)  # índices de los representantes más cercanos a cada centroide

# Obtener etiquetas de los representantes
y_representative_digits = y[idxs]  # Usamos y que ya contiene las etiquetas reales

# Para visualizar los representantes (opcional)
print(f"Número de representantes: {len(idxs)}")
print(f"Etiquetas de representantes: {y_representative_digits}")


# %%
from sklearn.cluster import KMeans
import numpy as np

# k óptimo encontrado (asegúrate de definir 'optimal_k' con el mejor valor encontrado previamente)
k = optimal_k_silhouette
kmeans = KMeans(n_clusters=k, random_state=42)
X_train_dist = kmeans.fit_transform(X)

# Obtener índices de muestras más cercanas a cada centroide
idxs = np.argmin(X_train_dist, axis=0)

# Obtener muestras representativas y sus etiquetas
X_representative_digits = X[idxs]
y_representative_digits = y[idxs]


# %% [markdown]
# Y entrenaremos un clasificados usando estos datos representativos.

# %%
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_representative_digits, y_representative_digits)


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Crear el vectorizador TF-IDF y ajustarlo al conjunto de entrenamiento
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')

# Asumiendo que X_train y X_test son datos de texto (como en el caso de Newsgroups)
X_train_transformed = vectorizer.fit_transform(newsgroups_train.data)  # Ajustar el vectorizador con los datos de entrenamiento

# Ajustar el modelo de regresión logística
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_train_transformed, newsgroups_train.target)

# Ahora transformamos el conjunto de prueba con el mismo vectorizador
X_test_transformed = vectorizer.transform(newsgroups_test.data)  # Aplicar la misma transformación al conjunto de prueba

# Evaluar la precisión
accuracy = log_reg.score(X_test_transformed, newsgroups_test.target)
print("Precisión en test:", accuracy)


# %% [markdown]
# entremanos con todo

# %%
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
%time log_reg.fit(X_train[:], y_train[:])
log_reg.score(X_test, y_test)

# %% [markdown]
# Esto pone de manifiesto que a la hora de entrenar modelos de ML no es tan importante la cantidad de datos, sino la calidad.
# 
# Ahora que tenemos un clasificador, podemos usarlo para anotar de manera automática el resto de los datos. Para ello asignaremos, en cada grupo, la misma etiqueta a todas las muestras que la muestra representativa.

# %%
# Propagate labels to the training data based on KMeans clusters
# Use shape[0] instead of len() for sparse matrices
y_train_propagated = np.empty(X.shape[0], dtype=int)
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]

# Print the distribution of propagated labels
unique_labels, counts = np.unique(y_train_propagated, return_counts=True)
print("Distribution of propagated labels:")
for label, count in zip(unique_labels, counts):
    print(f"Label {label}: {count} documents")

# %%
from sklearn.linear_model import LogisticRegression

# Create a new logistic regression model for the propagated labels
log_reg3 = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=10000, random_state=42)

# Fit the model using the TF-IDF transformed data and propagated labels
%time log_reg3.fit(X_train_transformed, y_train_propagated)

# Evaluate on the test set
log_reg3.score(X_test_transformed, newsgroups_test.target)

# %%
np.unique(y_train_propagated[:1000], return_counts=True)


