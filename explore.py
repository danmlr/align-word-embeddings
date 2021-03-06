"""This code should be used to load the objects required to explore the result of other computations. 
python -i explore.py
""" 


import numpy as np
import pandas as pd

from scipy.spatial import distance_matrix
from gensim.models import KeyedVectors
from ot.gromov import entropic_gromov_wasserstein
from ot.lp import free_support_barycenter

from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

from eval import *
import os 




epsilon=5e-4



languages = ['fr', 'en']
m = len(languages)
N = 10000
N0 = 5000
d = 300

embeddings = []
X = np.zeros((m, N, d))
C = np.zeros((m, N0, N0))

for i, l in tqdm(enumerate(languages)):
    embeddings.append(KeyedVectors.load_word2vec_format('../embeddings/cc.' + l + '.300.f10k.vec', binary=False))
    X[i] = embeddings[i].vectors
    C[i] = cosine_similarity(X[i, :N0], X[i, :N0])

l1 = 'fr'
l2 = 'en'
l1_l2_dict = pd.read_csv('dict/' + l1 + '_' + l2 + '_dict.csv', dtype = str).to_numpy()





# Pi=np.load('gromovWassertein_epsilon'+str(epsilon)+'.npz')['Pi']
# Pi3,Pi4=evalP10(embeddings[0],embeddings[1],l1_l2_dict,Pi)


print('Exemple : ','evalP10(embeddings[0],embeddings[1],l1_l2_dict,Pi)')
print('evalOptim(C[1],C[0],Pi)')     
