#from ot.gromov import entropic_gromov_wasserstein
import numpy as np 
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import cosine_similarity,manhattan_distances, rbf_kernel
from scipy.linalg import expm 



""" This files reimplements some functions that can be found in the POT package and that were designed based on the original paper : * Gabriel Peyré, Marco Cuturi, and Justin Solomon, "Gromov-Wasserstein averaging of kernel and distance matrices." International Conference on Machine Learning (ICML). 2016.*

This implementation corrects the expression for the gradient of the loss function by adding a forgotten term which brings crucial improvements for asymmetric cost matrices. 

More specifically : 
 - entropic_gromov_wasserstein should be replaced by entropic_gromov_wasserstein_corrected
 - gwggrad should be replaced by gwggrad_corrected 
 
For comparison purposes, original functions from the POT package are also included 
 """   
from ot.utils import list_to_array
from ot.backend import get_backend
from ot.bregman import sinkhorn

from ot.gromov import init_matrix ,gwloss, tensor_product




def entropic_gromov_wasserstein_corrected(C1, C2, p, q, loss_fun, epsilon,
                                max_iter=1000, tol=1e-9, verbose=False, log=False):
    
    C1, C2, p, q = list_to_array(C1, C2, p, q)
    nx = get_backend(C1, C2, p, q)

    T = nx.outer(p, q)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)
    constC2, _, _ = init_matrix(C1.T, C2.T, p, q, loss_fun)

    cpt = 0
    err = 1

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Tprev = T

        # compute the gradient 
        #----------------------->Modif here 
        tens = gwggrad(constC, hC1, hC2, T)+gwggrad(constC2, hC1.T, hC2.T, T)

        T = sinkhorn(p, q, tens, epsilon, method='sinkhorn')

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
                
            
            

        cpt += 1

    if log:
        log['gw_dist'] = gwloss(constC, hC1, hC2, T)
        return T, log
    else:
        return T


def gwggrad_corrected(constC, hC1, hC2, T,constC2):
    r"""Return the gradient for Gromov-Wasserstein

    The gradient is computed as described in Proposition 2 in :ref:`[12] <references-gwggrad>`

    Parameters
    ----------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    T : array-like, shape (ns, nt)
        Current value of transport matrix :math:`\mathbf{T}`

    Returns
    -------
    grad : array-like, shape (`ns`, `nt`)
           Gromov Wasserstein gradient


    .. _references-gwggrad:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    constC, hC1, hC2, T,constC2 = list_to_array(constC, hC1, hC2, T,constC2)
    
    nx = get_backend(constC, hC1, hC2, T)

    A = - nx.dot(
        nx.dot(hC1, T), hC2.T
    )
    tens = constC + A
   
    
    return  tens #[12] Prop. 2 misses more than a 2 factor 


    

def gwggrad(constC, hC1, hC2, T):
    r"""Return the gradient for Gromov-Wasserstein

    The gradient is computed as described in Proposition 2 in :ref:`[12] <references-gwggrad>`

    Parameters
    ----------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    T : array-like, shape (ns, nt)
        Current value of transport matrix :math:`\mathbf{T}`

    Returns
    -------
    grad : array-like, shape (`ns`, `nt`)
           Gromov Wasserstein gradient


    .. _references-gwggrad:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    constC, hC1, hC2, T = list_to_array(constC, hC1, hC2, T)
    nx = get_backend(constC, hC1, hC2, T)

    A = - nx.dot(
        nx.dot(hC1, T), hC2.T
    )
    tens = constC + A
    
    return 2 * tens  # [12] Prop. 2 misses a 2 factor
                              
                              



def entropic_gromov_wasserstein(C1, C2, p, q, loss_fun, epsilon,
                                max_iter=1000, tol=1e-9, verbose=False, log=False):
    r"""
    Returns the gromov-wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`

    The function solves the following optimization problem:

    .. math::
        \mathbf{GW} = \mathop{\arg\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} - \epsilon(H(\mathbf{T}))

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices
    - `H`: entropy

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p :  array-like, shape (ns,)
        Distribution in the source space
    q :  array-like, shape (nt,)
        Distribution in the target space
    loss_fun :  string
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.

    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two spaces

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    C1, C2, p, q = list_to_array(C1, C2, p, q)
    nx = get_backend(C1, C2, p, q)

    T = nx.outer(p, q)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    cpt = 0
    err = 1

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Tprev = T

        # compute the gradient
        tens = gwggrad(constC, hC1, hC2, T)

        T = sinkhorn(p, q, tens, epsilon, method='sinkhorn')

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
                
        
        cpt += 1

    if log:
        log['gw_dist'] = gwloss(constC, hC1, hC2, T)
        return T, log
    else:
        return T


## Experiment section


def H(M):
    """
    Returns the entropy of a postive matrix M 
    """
    return -np.sum(M*np.log(M))


def computeAttribution(CX,CY,alpha,marginal='uniform'):
    """
    
    This function returns the attribution matrices that minimize the Gromov Wasserstein loss function. 
    The optimization is done using both the original version of the POT implementation (1) and a corrected version of that implementation (2). 
    
    Inputs : 
    
    CX,CY : cost matrices. CX[i][j] (resp. CY[i][j]) should contain the value of the "distance" between point i and point j in the first (resp. second) point cloud. Both matrices are renormalized by a factor np.sqrt(np.std(CX)*np.std(CY)) for improved numerical stability
    
    alpha : The regularization parameter used in the optimization scheme. The epsilon from the original paper is deduced by renormalizing alpha by the entropy of the marginals. Typically, optimal values for alpha will be in the range [0.2 , 2] in most settings. 
    
    marginal (Optional) : The marginal law that defines the constraint for the optimization. By default, uniform marginals are used. 
    
    
    
    Outputs : 
    T : Attribution matrix for the Gromov Wasserstein problem computed using the POT implementaion
    T2 : Attribution matrix computed using the corrected gradient expression 
    
    
    """
    Npoints = CX.shape[0]
    
    if marginal=='zipf': # Zipf law    
        p = 1/np.arange(1,Npoints+1)
        p = p/np.sum(p)
    elif marginal=='random':
        p = np.random.normal(size=Npoints)
        p = p**2
        p = p/np.sum(p)
    else: # Uniform Law
        p = np.ones(Npoints)/Npoints
    q = p
    loss_fun = 'square_loss'
    
    scaling = np.sqrt(np.std(CX)*np.std(CY))
    CX = CX/scaling
    CY = CY/scaling
    # Epsilon resclaing in the case of square loss : 
    epsilon = alpha/(H(p)+H(q))
    
    constC, hC1, hC2 = init_matrix(CX, CY, p, q, loss_fun)
    
    
    
    lossOT = []
    lossOT_H = [] 
    
    lossOT_corrected = [] 
    lossOT_H_corrected = []
    
    T =entropic_gromov_wasserstein(CX,CY,p,q,epsilon=epsilon,loss_fun=loss_fun)
    T2 = entropic_gromov_wasserstein_corrected(CX,CY,p,q,epsilon=epsilon,loss_fun=loss_fun)
    
    # Final values of the loss function can be computed using the following code : 
    #finalLoss = gwloss(constC, hC1, hC2, T)
    #finalLoss2 = gwloss(constC, hC1, hC2, T2)
    return T,T2






## Experiment with matrix noise 

"""
We sample one hundred vectors (X) from the distribution N(0,I_300) and compute their cosine similarity matrix CX. We then sample a random 100x100 noise matrix M according to the standard normal distribution. 
We compute CY:=sigma*M+CX. We use uniform marginal laws.  
The quality of a solution T to the Gromov-Wassertein problem in that case can be well understood by examining the quantity Tr(T) (which we call precision). This number lies between 0 and 1 and corresponds to the total measure of correct matchings. 
We thus plot Tr(T) as a function of the noise intensity $\sigma$. We average our curves over 10 runs of the algorithm and use alpha=1. 
"""

import matplotlib.pyplot as plt 

plt.ylabel('Precision')
plt.xlabel('sigma')


def H(M):
    """
    Returns the entropy of a postive matrix M 
    """
    return -np.sum(M*np.log(M))



Npoints = 100
dim = 300



for _ in range(10):

    X = np.random.normal(size=(Npoints,dim))
    CX= cosine_similarity(X) 
    matrixNoise = np.random.normal(size=(Npoints,Npoints))
    
    noiseIntensity = 2*np.arange(0,12)/200
    #noiseIntensity = [0,0.2]
    
    pscore = np.zeros(len(noiseIntensity))
    pscoreCorrected = np.zeros(len(noiseIntensity))
    
    
    
    for numProblem,t in enumerate(noiseIntensity):
            
        CY = CX + t*matrixNoise
        
        T,T2 = computeAttribution(CX,CY,1)
        pscore[numProblem] += np.sum(np.diag(T))
        print('T',t,np.sum(np.diag(T)))
        pscoreCorrected[numProblem] += np.sum(np.diag(T2))
        print('T2',t,np.sum(np.diag(T2)))


pscore = pscore/10
pscoreCorrected=pscoreCorrected/10

plt.plot(noiseIntensity,pscore,'-b',label='GW baseline')
    
plt.plot(noiseIntensity,pscoreCorrected,'-c',label='GW with correction')


plt.legend()
plt.show()






