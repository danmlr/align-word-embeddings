import numpy as np
from tqdm import tqdm 

def eval_rot(l1, l2, l1_l2_dict, W):
    """ As input we use a list of translation pairs (same format as seed) and a rotation matrix W 
        Output : percentage of success P@10 
    """
    success = 0
    for l1_word,l2_word in l1_l2_dict: 
        x = l1.word_vec(l1_word)
        translates = l2.similar_by_vector(W@x)
        for transword,_ in translates:
            if transword==l2_word:
                success+=1 
                break 
    return 100*success/len(l1_l2_dict)

def eval_perm(l1, l2, l1_l2_dict, P):
    """P[i, j] donne la probabilité que le mot i de l1 soit traduit par le mot j de l2"""

    nb_tested_words = 0
    success = 0

    for l1_word, l2_word in l1_l2_dict:

        i = l1.vocab[l1_word].index
        
        if i >= P.shape[0]:
            break
        
        nb_tested_words += 1

        indices = np.argsort(P[i])[-10:]
        translates = [l2.index2word[j] for j in indices]

        for transword in translates:
            if transword == l2_word:
                success += 1
                break

    return 100 * success / nb_tested_words



def evalP10(l1, l2, l1_l2_dict, Pi):
	""" Evalue différentes P@10 pour une permutation de supervision donnée """
	print('P@10 Permutation : ',eval_perm(l1, l2, l1_l2_dict, Pi.T))
	u,_,vh = np.linalg.svd(Pi) 
	Pi2 = u@vh
	print('P@10 Permutation après projection sur orthogonales: ',eval_perm(l1, l2, l1_l2_dict, Pi2.T))
	N0 = Pi.shape[0]
	u, _, vh = np.linalg.svd(l2.vectors[:N0].T @ Pi @ l1.vectors[:N0])
	Q = u@vh 
	print('P@10 Rotation : ',eval_rot(l1, l2, l1_l2_dict, Q))
	u, _, vh = np.linalg.svd(l2.vectors[:N0].T @ Pi2 @ l1.vectors[:N0])
	Q2 = u@vh
	print('P@10 Rotation après projection : ',eval_rot(l1, l2, l1_l2_dict, Q2))



def entropy(M):
	return -np.sum(M*np.log(M))

def OTGW(C1,C2,Pi):
	""" Assuming square loss and bistochastic constraints  """
	N1 = C1.shape[0]
	N2 = C2.shape[0] 
	p = np.ones((N1,1))
	q = np.ones((N2,1))
	A = C1**2@p@np.ones((N2,1)).T
	B = np.ones((N1,1))@q.T@(C2**2).T
	C = - 2*C1@Pi@C2.T
	LxT = A+B+C
	return np.trace(LxT@Pi.T)



def evalOptim(C1,C2,T):
	print('OT GW loss : ',OTGW(C1,C2,T))
	print('Entropy : ',entropy(T)) 



def projBistoch(C1,C2,Pi):
	N1 = C1.shape[0]
	N2 = C2.shape[0]
	p = np.ones((N1,1))
	q = np.ones((N2,1))
	A = C1**2@p@np.ones((N2,1)).T
	B = np.ones((N1,1))@q.T@(C2**2).T
	constC=A+B
	hC1 = C1
	hC2 = 2*C2
	tens = gwggrad(constC, hC1, hC2, T)
	T = sinkhorn(p, q, tens, epsilon, method='sinkhorn')
	return T
