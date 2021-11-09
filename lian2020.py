import numpy as np
from scipy.spatial import distance_matrix
from gensim.models import KeyedVectors
from ot.gromov import entropic_gromov_wasserstein
from ot.lp import free_support_barycenter

from tqdm import tqdm

languages = ['fr', 'en', 'de', 'es']
m = len(languages)
N = 10000
N0 = 1000
NY = 10000
d = 300
Niter = 10

embeddings = []
X = np.zeros((m, N, d))
C = np.zeros((m, N0, N0))
Pi = np.zeros((m, N0, N0))
Q = np.zeros((m, d, d))
Q[0] = np.eye(d)
WX = np.ones((m, N)) / N

for i, l in tqdm(enumerate(languages)):
    embeddings.append(KeyedVectors.load_word2vec_format('../embeddings/cc.' + l + '.300.f10k.vec', binary=False))
    X[i] = embeddings[i].vectors
    C[i] = distance_matrix(X[i, :N0], X[i, :N0])
    if i > 0:
        Pi[i] = entropic_gromov_wasserstein(C[i], C[0], np.ones(N0) / N0, np.ones(N0) / N0, 'square_loss', epsilon=5e-4)
        u, _, vh = np.linalg.svd(X[i, :N0].T @ Pi[i] @ X[0, :N0])
        Q[i] = u @ vh
        X[i] = X[i] @ Q[i]

Y = X[0]
WY = np.ones(NY) / NY

for iter in tqdm(range(Niter)):
    print('##############')
    print('Iteration : ', iter)
    Y = free_support_barycenter(X, WX, Y, WY, np.ones(m) / m)
    for i in range(m):
        M = distance_matrix(X[i], Y)
        Pi[i] = ot.sinkhorn(WX[i], WY, M, 1)
        u, _, vh = np.linalg.svd(X[i].T @ Pi[i] @ Y)
        Q[i] = Q[i] @ u @ vh
        X[i] = X[i] @ u @ vh
        print('-------------')
        print(languages[i])
        print(np.linalg.norm(X[i] @ Q[i] - Pi[i] @ Y))

np.savez_compressed('lian2020_res', Q=Q, Pi = Pi)
