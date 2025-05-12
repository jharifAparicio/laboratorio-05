# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juansensio/blog/blob/master/096_ml_unsupervised/096_ml_unsupervised.ipynb)

# %% [markdown]
# ### Aprendizaje Activo

# %% [markdown]
# El aprendizaje activo (o *Active Learning*) consiste en entrenar modelos de ML de manera iterativa, incluyendo en cada iteración nuevas muestras al dataset focalizando en ejemplos en loa que el modelo tenga más problemas.

# %%
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Cargar el dataset
newsgroups = fetch_20newsgroups(subset='train')
X_train, y_train = newsgroups.data, newsgroups.target
newsgroups = fetch_20newsgroups(subset='test')
X_test, y_test = newsgroups.data, newsgroups.target

# Vectorización del texto
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)

# Entrenar el modelo de regresión logística
log_reg3 = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=10000, random_state=42)
log_reg3.fit(X_train_tfidf, y_train)

# Obtener las probabilidades de las clases
probas = log_reg3.predict_proba(X_train_tfidf)

# Obtener los índices de la clase con mayor probabilidad
labels_ixs = np.argmax(probas, axis=1)

# Obtener las probabilidades asociadas a esas clases
labels = np.array([proba[ix] for proba, ix in zip(probas, labels_ixs)])

# Ordenar los índices de las probabilidades (de menor a mayor)
sorted_ixs = np.argsort(labels)

# Mostrar las 10 muestras con la probabilidad más baja (menos confiables)
print("10 muestras menos confiables (con menor probabilidad):")
for ix in sorted_ixs[:10]:
    print(f"Índice: {ix}, Probabilidad: {labels[ix]:.4f}")


# %%
import matplotlib.pyplot as plt

# Obtener los k índices de las muestras más inciertas
k = 10  # Puedes cambiar este valor si deseas más o menos ejemplos
# Usamos solo los índices de las primeras 1000 muestras para coincidir con la selección de datos
lowest_documents = [X_train[i] for i in sorted_ixs[:k] if i < 1000]

# Mostrar los documentos seleccionados
plt.figure(figsize=(10, 4))
for index, doc in enumerate(lowest_documents):
    plt.subplot(k // 10, 10, index + 1)
    plt.text(0.5, 0.5, doc[:300] + '...', ha='center', va='center', wrap=True)  # Solo mostrar una parte del documento
    plt.axis('off')
plt.show()


# %%
# X_train is a list object, let's check its length
print(f"Length of X_train[:1000]: {len(X_train[:1000])}")

# Print shape of X_train_tfidf which is a sparse matrix 
print(f"Shape of X_train_tfidf[:1000]: {X_train_tfidf[:1000].shape}")

# Check y_train2 shape
print(f"Shape of y_train2: {y_train.shape}")


# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# Medir el tiempo de ejecución con un comando mágico en Jupyter Notebook
%time log_reg5.fit(X_train_transformed[:1000], y_train2)

# Evaluar el modelo
accuracy = log_reg5.score(X_test_transformed, y_test)
print(f"Precisión en test: {accuracy}")



# %%
# En Jupyter Notebook, el código se puede ejecutar como sigue:
log_reg5 = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)

# Preparar y usar datos adecuados - utiliza la versión transformada para mantener consistencia
# También asegurarse de que estamos usando los 1000 primeros datos del conjunto de entrenamiento
X_train_subset_tfidf = X_train_tfidf[:1000]
y_train2 = y_train[:1000]

# Medir el tiempo de ejecución de la línea de código
%time log_reg5.fit(X_train_subset_tfidf, y_train2)

# Evaluar el modelo en el conjunto de prueba transformado
accuracy = log_reg5.score(X_test_tfidf, y_test)
print(f"Precisión en test: {accuracy}")


# %% [markdown]
# Podemos repetir el proceso tantas veces como haga falta hasta llegar a las prestaciones requeridas.


