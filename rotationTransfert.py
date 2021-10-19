
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


Nseed = 2000 
seed1 = arr[np.random.choice(len(arr),Nseed)]
W1 = rot(seed1)
seed2 = arr[np.random.choice(len(arr),Nseed)]
W2 = rot(seed2)
seed3 = arr[np.random.choice(len(arr),Nseed)]

w,v = np.linalg.eig(W2@W1.T) 
wabs = np.abs(w) 
warg = np.angle(w)


print('Should be zero if v is orthogonal',np.linalg.norm(v@np.conjugate(v.T)-np.eye(300)))

vinvW1 =np.conjugate(v.T)@W1

Nt = 9
P=np.zeros((4,Nt))
for i in range(Nt):
	t = i/Nt 
	W = v@np.diag(wabs*np.exp(1j*t*warg))@vinvW1
	W = np.real(W)
	p1 = eval(seed1,W)
	p2 = eval(seed2,W) 
	p3 = eval(seed3,W) 
	P[1,i] = p1 
	P[2,i] = p2 
	P[3,i] = p3 
	print(str(p1) +','+ str(p2)+','+str(p3)) 
 
np.savetxt("Pchemin.csv", P, delimiter=",")



