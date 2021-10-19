import pandas as pd
import gensim.downloader
import numpy as np 

en_vectors = gensim.downloader.load('word2vec-google-news-300')
ru_vectors = gensim.downloader.load('word2vec-ruscorpora-300')

df =pd.read_csv('ru_en_dict.csv', dtype = str)
arr = df.to_numpy()


def rot(seed):
	
	X = np.array([ru_vectors.word_vec(x) for x in seed[:,0]]).T 
	Z = np.array([en_vectors.word_vec(x) for x in seed[:,1]]).T

	#On cherche à minimiser la norme de WX-Z, SVD de XZ^T = U S V^T donne W = VU^T 
	u,_,vh = np.linalg.svd(X@Z.T)

	return vh.T@u.T 


def rotSim(W1,W2):
	""" Returns the rotation similarity between two learned matrices. The first value is averaged rotation angle 
	    in absolute value (in degrees). The second value is the L2 norm of the angle vector (radians) 
	""" 

	#Compute the angle distance between the two matrices 
	w,_ = np.linalg.eig(W1.T@W2)
	angles = 180/np.pi*np.angle(w)
	angles.sort()
	return np.mean(np.abs(angles)), np.linalg.norm(np.pi/180*angles)

# Assess translation performances ru->en 
print('Training set') 
def eval(testSet,W):
	""" As input we use a list of translation pairs (same format as seed) and a rotation matrix W 
	    Output : percentage of success P@10 
	"""
	success = 0
	for ru_word,en_word in testSet: 
		x = ru_vectors.word_vec(ru_word)
		translates = en_vectors.similar_by_vector(W@x)
		for transword,_ in translates:
			if transword==en_word:
				success+=1 
				break 
	return 100*success/len(testSet)  

Nrepet = 10
Nseeds = [4000,5000,6000,7000,8000,8300] 

P11 = np.zeros((len(Nseeds),2*Nrepet))
P12 = np.zeros((len(Nseeds),2*Nrepet))

MeanAngle = np.zeros((len(Nseeds),Nrepet))
NormL2  = np.zeros((len(Nseeds),Nrepet))

for i,Nseed in enumerate(Nseeds): 
	for j in range(Nrepet):
		print('------------------------------------------')
		print('Nseed= '+str(Nseed)+' '+str(j)+'/'+str(Nrepet)) 
		seed1 = arr[np.random.choice(len(arr),Nseed)]
		W1 = rot(seed1)
		seed2 = arr[np.random.choice(len(arr),Nseed)]
		W2 = rot(seed2)

		p11 = eval(seed1,W1)
		p21 = eval(seed2,W1)
		p12 = eval(seed1,W2) 
		p22 = eval(seed2,W2) 
		print('\n P@10 computations on training set \n') 
		print('seed1W1 : '+str(p11) + ' | '+ 'seed1W2 : '+str(p12) )
		print('seed1W1 : '+str(p21) + ' | '+ 'seed2W2 : '+str(p22) )

		meanAngle, normL2 = rotSim(W1,W2)
		print('Angle moyen : ' + str(meanAngle)) 
		print('L2-norm : ' +str(normL2))    
		
		P11[i,2*j] = p11
		P11[i,2*j+1]= p22
		P12[i,2*j] = p12
		P12[i,2*j+1] = p21

		MeanAngle[i,j] = meanAngle
		NormL2[i,j] = normL2


import sys
np.set_printoptions(threshold=sys.maxsize)
print('\n\nP11')
print(P11)
print('\n\nP12')
print(P12)

print('\n\nMeanAngle') 
print(MeanAngle)
print('\n\nNormL2')
print(NormL2)  


np.savetxt("P11.csv", P11, delimiter=",")
np.savetxt("P12.csv", P12, delimiter=",")

np.savetxt("MeanAngle.csv", MeanAngle, delimiter=",")
np.savetxt("NormL2.csv", NormL2, delimiter=",")

np.savetxt('Nseeds.csv',Nseeds,delimiter=",") 
