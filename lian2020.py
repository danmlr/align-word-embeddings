import numpy as np
import pandas as pd

from scipy.spatial import distance_matrix
from gensim.models import KeyedVectors
from ot.gromov import entropic_gromov_wasserstein
from ot.lp import free_support_barycenter

from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

from eval import eval_perm

languages = ['fr', 'en']
m = len(languages)
N = 10000
N0 = 5000
NY = 10000
d = 300
Niter = 10

def gromov_wasserstein(epsilon):
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
        C[i] = cosine_similarity(X[i, :N0], X[i, :N0])
        if i > 0:
            Pi[i],log = N0 * entropic_gromov_wasserstein(C[i], C[0], np.ones(N0) / N0, np.ones(N0) / N0, 'square_loss', epsilon=epsilon,log=True)
            u, _, vh = np.linalg.svd(X[i, :N0].T @ Pi[i] @ X[0, :N0])
            Q[i] = u @ vh
            X[i] = X[i] @ Q[i]

    l1 = 'fr'
    l2 = 'en'

    l1_l2_dict = pd.read_csv('dict/' + l1 + '_' + l2 + '_dict.csv', dtype = str).to_numpy()

    permP10 = eval_perm(embeddings[0], embeddings[1], l1_l2_dict, Pi[1].T)
    rotP10 = eval_rot(embeddings[0], embeddings[1], l1_l2_dict, Q[1])
    np.savez_compressed('gromovWassertein_epsilon'+str(epsilon),epsilon=epsilon,permP10=permP10,rotP10=rotP10, Q=Q[1], Pi = Pi[1],log=log)

for epsilon in [2e-3,5e-4,1e-3]:
	gromov_wasserstein(epsilon)	





# Y = X[0]
# WY = np.ones(NY) / NY

# for iter in tqdm(range(Niter)):
#     print('##############')
#     print('Iteration : ', iter)
#     Y = free_support_barycenter(X, WX, Y, WY, np.ones(m) / m)
#     for i in range(m):
#         M = distance_matrix(X[i], Y)
#         Pi[i] = ot.sinkhorn(WX[i], WY, M, 1)
#         u, _, vh = np.linalg.svd(X[i].T @ Pi[i] @ Y)
#         Q[i] = Q[i] @ u @ vh
#         X[i] = X[i] @ u @ vh
#         print('-------------')
#         print(languages[i])
#         print(np.linalg.norm(X[i] @ Q[i] - Pi[i] @ Y))

# np.savez_compressed('lian2020_res', Q=Q, Pi = Pi)
