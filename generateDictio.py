import numpy as np
import pandas as pd

from deep_translator import GoogleTranslator
import gensim.downloader

en_vectors = gensim.downloader.load('word2vec-google-news-300')
ru_vectors = gensim.downloader.load('word2vec-ruscorpora-300')

ru_en_dict = []

ru_indices = np.random.choice(15000, 10000)

compteur = 0 
for ru_index in ru_indices:
	ru_word = ru_vectors.index2word[ru_index]
	en_word = GoogleTranslator(source='ru', target='en').translate(ru_word.split('_')[0])
	compteur += 1 
	if compteur%10==0:
		print('Etape :' + str(compteur))  
	try:
		en_vectors.word_vec(en_word)
		ru_en_dict.append([ru_word, en_word])
	except KeyError:
		print('Bad translation :' + en_word)

a = pd.DataFrame(ru_en_dict)
a.to_csv('ru_en_dict.csv', header=False, index=False)
