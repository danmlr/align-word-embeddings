"""
This code generates bilingual dictionnaries between the languages l1 and l2 by using Google Translate. 
The word embeddings are required in both languages in order to make sure that every entry in the dictionnary can be found by the method word_vec. 
This code attempts to translate the first 10k words given by the l1-embedding. Because some translations will fail, you should expect the dictionnary to have strictly less than 10k items. 
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from gensim.models import KeyedVectors

from deep_translator import GoogleTranslator
from deep_translator.exceptions import NotValidPayload

l1 = 'en'
l2 = 'es'

l1_vectors = KeyedVectors.load_word2vec_format('../embeddings/cc.' + l1 + '.300.f10k.vec', binary=False)
l2_vectors = KeyedVectors.load_word2vec_format('../embeddings/cc.' + l2 + '.300.f10k.vec', binary=False)

print('Models loaded')

l1_l2_dict = []

l1_indices = np.arange(10000)

compteur = 0 
for l1_index in tqdm(l1_indices):
	l1_word = l1_vectors.index2word[l1_index]
	try:
		l2_word = GoogleTranslator(source=l1, target=l2).translate(l1_word)
	except NotValidPayload:
		continue
	compteur += 1 
	try:
		l2_vectors.word_vec(l2_word)
		l1_l2_dict.append([l1_word, l2_word])
	except KeyError:
		pass
		# print('Bad translation : ' + str(l2_word))

a = pd.DataFrame(l1_l2_dict)
a.to_csv('dict/' + l1 + '_' + l2 + '_dict.csv', header=False, index=False)
