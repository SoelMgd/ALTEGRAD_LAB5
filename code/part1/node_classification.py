"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk


# Loads the karate network
G = nx.read_weighted_edgelist('/content/ALTEGRAD_LAB5/code/data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('/content/ALTEGRAD_LAB5/code/data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network

##################
color_map = ['red' if label == 0 else 'blue' for label in y]

plt.figure(figsize=(8, 8))
nx.draw_networkx(G, node_color=color_map, with_labels=True, node_size=500, font_size=10)
plt.title("Karate Club Network", fontsize=16)
plt.savefig('KarateNetwork.png') 
plt.show()
##################


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions


##################
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification accuracy with DeepWalk embeddings: {accuracy:.2f}")
##################


############## Task 8
# Generates spectral embeddings

##################

# Normalized graph Laplacian
A = nx.adjacency_matrix(G).astype(float)  # Adjacency matrix
degrees = np.array(A.sum(axis=1)).flatten()
D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
L_rw = eye(n) - D_inv_sqrt @ A @ D_inv_sqrt  # Normalized Laplacian

# Compute the two smallest eigenvectors of the Laplacian
eigenvalues, eigenvectors = eigs(L_rw, k=2, which='SM')

# Use the eigenvectors as spectral embeddings
spectral_embeddings = np.real(eigenvectors)

X_train_spectral = spectral_embeddings[idx_train, :]
X_test_spectral = spectral_embeddings[idx_test, :]


clf_spectral = LogisticRegression(max_iter=1000, random_state=42)
clf_spectral.fit(X_train_spectral, y_train)
y_pred_spectral = clf_spectral.predict(X_test_spectral)
accuracy_spectral = accuracy_score(y_test, y_pred_spectral)

print(f"Classification accuracy with spectral embeddings: {accuracy_spectral:.2f}")
##################
