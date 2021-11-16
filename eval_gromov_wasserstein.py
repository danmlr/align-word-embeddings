import numpy as np

from gensim.models import KeyedVectors
from ot.gromov import entropic_gromov_wasserstein

from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

from time import time

languages = ['fr', 'de']
m = len(languages)
N = 10000
N0 = 5000
d = 300
Niter = 10

def gromov_wasserstein(epsilon, N0):

    embeddings = []
    X = np.zeros((m, N, d))
    C = np.zeros((m, N0, N0))
    Pi = np.zeros((m, N0, N0))
    Q = np.zeros((m, d, d))
    Q[0] = np.eye(d)

    for i, l in tqdm(enumerate(languages)):

        embeddings.append(KeyedVectors.load_word2vec_format('../embeddings/cc.' + l + '.300.f10k.vec', binary=False))

        X[i] = embeddings[i].vectors
        C[i] = cosine_similarity(X[i, :N0], X[i, :N0])

        if i > 0:

            # Compute Gromov Wasserstein between i and 0
            t0 = time()
            Pi[i]= N0 * entropic_gromov_wasserstein(C[i], C[0], np.ones(N0) / N0 , np.ones(N0) / N0, 'square_loss', epsilon=epsilon, verbose=True)
            dT = time() - t0

            # Compute generalizing rotation
            u, _, vh = np.linalg.svd(X[i, :N0].T @ Pi[i] @ X[0, :N0])
            Q[i] = u @ vh

            # Save result
            fname = 'gromovWassertein_' + l + '-' + languages[0] + '_epsilon' + str(epsilon) + '_N0' + str(N0)
            np.savez_compressed(fname, dT=dT, epsilon=epsilon, N0=N0, Q=Q[i], Pi=Pi[i])


for epsilon, N0 in [(2e-3,5000), (1e-3,5000), (5e-4,5000), (2e-4,5000)]:
	gromov_wasserstein(epsilon, N0)
