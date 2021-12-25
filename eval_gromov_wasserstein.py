""" 
This file contains code to compute Gromov Wasserstein alignment between word embedding spaces in multiple languages. 
""" 


import numpy as np

from gensim.models import KeyedVectors
from ot.gromov import entropic_gromov_wasserstein

from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

from time import time





def gromov_wasserstein(epsilon, N0=2000,languages = ['fr', 'en' ,'es'], N=10000):
    """
    This function uses the Gromov Wasserstein minimization problem to compute attribution matrices between words in different languages. Isometries that map spaces between different languages are also computed Results are saved in a file named 'gromovWassertein...'
    
    Input : 
        epsilon : Regularization parameter for the optimization 
        N0 : Number of words used by the Gromov Waserstein computation. The first N0 in X are selected 
        languages :  The first language is used as the pivot 
        N : Number of words in the embeddings. 
        
    """ 
    
    m = len(languages)
    d = 300 # Dimension of the embedding space 
    
    embeddings = []
    X = np.zeros((m, N, d)) # X[l][i] will contain the raw embedding of the i-th word in language l 
    C = np.zeros((m, N0, N0)) 
    Pi = np.zeros((m, N0, N0))
    Q = np.zeros((m, d, d)) #Q[l] contains the isometry matrix that maps the space of language 0 into that of language l 
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



## Experiment - Compute Gromov Wasserstein attribution matrices for different values of N0 and epsilon

for epsilon, N0 in [(2e-3,5000), (1e-3,5000), (5e-4,5000), (2e-4,5000)]:
    gromov_wasserstein(epsilon, N0)
