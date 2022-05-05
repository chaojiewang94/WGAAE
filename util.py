import numpy as np
import time
from scipy.spatial import distance
from scipy.stats import bernoulli, poisson, uniform

real_min = np.float64(2.2e-10)

class Empty():
    pass

def Normlization(X, mode, dim):

    # X: V*N
    # mode: L1, L2, Gaussian
    if mode == 'L1':
        col_sum = np.sum(X, dim, keepdims=True)
        return X / col_sum

    elif mode == 'L2':
        col_sum = np.sum(X*X, dim, keepdims=True)
        return X / np.sqrt(col_sum)

    elif mode == 'Gaussian':
        col_mean = np.mean(X, dim, keepdims=True)
        col_std = np.std(X, dim, keepdims=True)
        return (X - col_mean) / col_std

def Cosine_Simlarity(A, B):

    # A: N*D, B: N*D
    [N, D] = A.shape
    inter_product = np.matmul(A, np.transpose(B))  # N*N
    len_A = np.sqrt(np.sum(A*A, axis=1, keepdims=True))
    len_B = np.sqrt(np.sum(B*B, axis=1, keepdims=True))
    len_AB = np.matmul(len_A, np.transpose(len_B)) + real_min
    cos_AB = inter_product / len_AB
    cos_AB[(np.arange(N), np.arange(N))] = 1
    cos_AB[cos_AB>0.2] = 1 # R8 : >0.8992
    cos_AB[cos_AB<0.3] = 0
    return cos_AB

def build_graph(data):

    """
    :param data: matrix of v*n
    :return: calculate graph of n*n by cos distance,range[0,1]
    """
    import cv2
    V,N = data.shape
    A = np.ones((N,N))
    print('build graph ...')
    time1 = time.time()
    for i in range(N):
        for j in range(i+1,N):
            A[i,j] = bernoulli.rvs(distance.cosine(1.0*data[:,i],1.0*data[:,j]))
            A[j,i] = A[i,j]
    time2 = time.time()
    print('cost time: %3.2f s'%(time2-time1))
    return A

def sample_M(A,U,theta):
    """
    :param A: N*N , 0 or 1
    :param U: K*1
    :param theta: K*N
    :return: M_ij matrix
    """
    def reject_sampler(poisson_para):
        """
        :param poisson_para: N*N
        :return:
        """
        N = poisson_para.shape[0]
        m_ij = np.zeros((N,N))
        for i in range(N):
            for j in range(i+1,N):
                if poisson_para[i,j] <1:
                    while True:
                        n = poisson.rvs(poisson_para[i,j])
                        u = uniform.rvs()
                        if u < 1.0/(n+1):
                            m_ij[i,j] = n+1
                            m_ij[j,i] = n+1
                            break
                else :
                    while True:
                        n = poisson.rvs(poisson_para[i,j])
                        if n >= 1:
                            m_ij[i,j] = n
                            m_ij[j,i] = n
                            break
        return m_ij

    lamda_para = np.dot(theta.T,np.diag(U[:,0]))
    lamda_para = np.dot(lamda_para,theta) # N*N
    m_ij = reject_sampler(lamda_para)
    return A*m_ij