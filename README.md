# Multilingual alignment of word embeddings 

Authors : Gonzague de Carpentier - Dan Meller 
This code was written during a research project in Natural Language Processing at Ecole Polytechnique.  
We wish to thank François Yvon (Limsi/CNRS) for his supervision.  



## How to add word embeddings 
Word embeddings in text format need to be placed in a directory named *embeddings* located **outside** of this git repository. 
The name format of the files need to follow the convention : *'cc'.language.dimension.'f'NumberOfWords.'vec'* 
Example : *cc.fr.300.f10k.vec* 
Such embeddings can be found directly on the fastText webpage : https://fasttext.cc/docs/en/crawl-vectors.html 


## What can be found in this repository ? 

### pot_gromov_wasserstein.py 
 This files reimplements some functions that can be found in the POT package and that were designed based on the original paper : * Gabriel Peyré, Marco Cuturi, and Justin Solomon, "Gromov-Wasserstein averaging of kernel and distance matrices." International Conference on Machine Learning (ICML). 2016.*

This implementation corrects the expression for the gradient of the loss function by adding a forgotten term which brings crucial improvements for asymmetric cost matrices. 

More specifically : 
 - entropic_gromov_wasserstein should be replaced by entropic_gromov_wasserstein_corrected
 - gwggrad should be replaced by gwggrad_corrected 
 
For comparison purposes, original functions from the POT package are also included


### rotation.py 
 This file contains some code that computes the solution to the Procrustes problem for a given seed. 
 It also allows to compute the dissimilarity between two isometry using a rotation reduction. 
 We include an expriment section which can be used to analyze the variability of the solution to the Procrustes problem when changing the seed. 

### eval.py 

 This files provides functions that evaluate the quality of word embeddings alignments using bilingual word translations. 


###  eval_gromov_wasserstein.py
 This file contains code to compute Gromov Wasserstein alignment between word embedding spaces in multiple languages. 

### explore.py 

 This code should be used to load the objects required to explore the result of other computations. 
 ``python -i explore.py``

### generate_dictio.py 

 This code generates bilingual dictionnaries between the languages l1 and l2 by using Google Translate. 
 The word embeddings are required in both languages in order to make sure that every entry in the dictionnary can be found by the method word_vec. 
 This code attempts to translate the first 10k words given by the l1-embedding. Because some translations will fail, you should expect the dictionnary to have strictly less than 10k items. 


