import numpy as np
from gensim.models import KeyedVectors

from tomaster import tomato

l1 = 'fr'
l1_vectors = KeyedVectors.load_word2vec_format('../embeddings/cc.' + l1 + '.300.f10k.vec', binary=False)

res = tomato(points=l1_vectors.vectors, k=5, tau=0.1)

np.savetxt("res/clusters_fr_10k.csv", res, delimiter=",")
