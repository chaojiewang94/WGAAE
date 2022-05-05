#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 17:15:34 2017

@author: Chaojie
"""
import numpy as np
import numpy.ctypeslib as npct

#from sample import Sampler
import ctypes  
from ctypes import *
realmin = 2.2e-308

array_2d_double = npct.ndpointer(dtype=np.double,ndim=2,flags='C')
array_1d_double = npct.ndpointer(dtype=np.double,ndim=1,flags='C')
array_int = npct.ndpointer(dtype=np.int32,ndim=0,flags='C')
ll = ctypes.cdll.LoadLibrary   

Multi_lib = ll("./libMulti_Sample.so")
Multi_lib.Multi_Sample.restype = None
Multi_lib.Multi_Sample.argtypes = [array_2d_double,array_2d_double,array_2d_double,array_2d_double,array_2d_double, c_int, c_int, c_int]
Multi_lib.Multi_Input.restype = None
Multi_lib.Multi_Input.argtypes = [array_2d_double,array_2d_double,array_2d_double,array_2d_double,array_2d_double, c_int, c_int]


Crt_Multi_lib = ll("./libCrt_Multi_Sample.so")
Crt_Multi_lib.Crt_Multi_Sample.restype = None
Crt_Multi_lib.Crt_Multi_Sample.argtypes = [array_2d_double,array_2d_double,array_2d_double,array_2d_double,array_2d_double, c_int, c_int, c_int]


Crt_lib =  ll("./libCrt_Sample.so")
Crt_lib.Crt_Sample.restype = None
Crt_lib.Crt_Sample.argtypes = [array_2d_double,array_2d_double, array_2d_double, c_int,c_int]

# ====================== CUDA Initial ======================#
import pycuda.curandom as curandom
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

mod = SourceModule("""
#include <stdio.h>
__device__ int cudarand(long long seed)
{
    long long temp=(48271 * seed + 0) % 2147483647;
    return temp;
}

__global__ void Multi_Sample(float *randomseed, int* scale, int* X_value, int* X_rows, int* X_cols, float* Phi, float* Theta, float* XVK, float* XKJ)    //
{
    const int V = scale[0];
    const int K = scale[1];
    const int J = scale[2];
    const int n_x = scale[3];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seed = randomseed[idx]*2147483647.0;

    if (idx < n_x)
    {
        float cumsum = 0.0;
        float sum = 0.0;
        int v = X_rows[idx];
        int j = X_cols[idx];

        for (int k = 0; k < K; k++)
        {
            sum += Phi[v*K + k] * Theta[k*J + j];
        }

        for (int token = 0; token<X_value[idx]; token++)
        {
            seed = cudarand(seed);
            float probrnd = ((float)(seed) / 2147483647.0) * sum;

            int Embedding_K = K-1;
            float sumprob = 0.0;
            for (int k = 0; k < K; k++)
            {
                cumsum = Phi[v*K + k] * Theta[k*J + j];
                if (sumprob+cumsum>=probrnd)
                {
                    Embedding_K=k;
                    break;
                }
                sumprob += cumsum;
            }
            atomicAdd(&XVK[v*K + Embedding_K], float(1));
            atomicAdd(&XKJ[Embedding_K*J + j], float(1));
        }
    }
}
""")

# check correct
def Calculate_pj(c_j,T):  #c_j 0-T    
    p_j = []
    N = c_j[1].size
    p_j.append((1-np.exp(-1))*np.ones([1,N]))     # test same as Matlab exp(-1)
    p_j.append(1/(1+c_j[1]))
    
    for t in [i for i in range(T+1) if i>1]:    # only T>=2 works
        tmp = -np.log(np.maximum(1-p_j[t-1],realmin));
        p_j.append(tmp/(tmp+c_j[t]))
        
    return p_j

#def Calculate_pj_2(c_j,p_j,T,p_j_key):  #c_j 0-T    
#    
#    N = c_j[p_j_key].size
#    current_layer = p_j_key[0]
#    shared_layer = p_j_key[2]
#    fp_j_key = str(current_layer-1) + '_' + str(shared_layer)
#    
#    if current_layer == 1: 
#        return (1-np.exp(-1))*np.ones([1,N])
#    elif current_layer == 2:
#        return (1 / c_j[p_j_key])
#    else:
#        tmp = -np.log(np.maximum(1-p_j[fp_j_key],realmin));
#        return (tmp/(tmp+c_j[p_j_key]))
        
#    p_j = []
#    
#    p_j.append((1-np.exp(-1))*np.ones([1,N]))     # test same as Matlab exp(-1)
#    p_j.append(1/(1+c_j[1]))
#    
#    for t in [i for i in range(T+1) if i>1]:    # only T>=2 works
#        tmp = -np.log(np.maximum(1-p_j[t-1],realmin));
#        p_j.append(tmp/(tmp+c_j[t]))
#        
#    return p_j
        
def Multrnd_Matrix(X_t,Phi_t,Theta_t):
    V = X_t.shape[0]
    J = X_t.shape[1]
    K = Theta_t.shape[0]
    X_t = np.array(X_t, order='C').astype('double')
    Xt_to_t1_t = np.zeros([K, J], order='C').astype('double')
    WSZS_t = np.zeros([V, K], order='C').astype('double')
    
    Multi_lib.Multi_Sample(X_t, Phi_t, Theta_t, WSZS_t, Xt_to_t1_t, V, K, J)
#    V = X_t.shape[0]
#    J = X_t.shape[1]
#    K = Theta_t.shape[0]
#    a = Sampler() 
#    Xt_to_t1_t = np.zeros([K,J])
#    WSZS_t = np.zeros([V,K])
#    Embedding = np.zeros(K,dtype='uint32')
#    Augment_Matrix = np.zeros([V,K,J],dtype='uint32')
#    
#    for n in range(Xt_xy.shape[1]):
#        v = Xt_xy[0,n]
#        j = Xt_xy[1,n]
#        Pro =  (Phi_t[v,:] *Theta_t[:,j])/(Phi_t[v,:] *Theta_t[:,j]).sum(0)
#        a.multinomial(X_t[v,j],Pro,Embedding)   ## output to embedding
#        Augment_Matrix[v,:,j] = Embedding
#    Xt_to_t1_t[:,:] = Augment_Matrix.sum(0);
#    WSZS_t[:,:] = Augment_Matrix.sum(2);
    return Xt_to_t1_t, WSZS_t


def Multrnd_Matrix_GPU(X_t, Phi_t, Theta_t):
    func = mod.get_function('Multi_Sample')
    [V, J] = X_t.shape
    K = Theta_t.shape[0]

    # [X_t_rows, X_t_cols] = np.where(np.ones_like(X_t))
    # X_t_values = X_t[(X_t_rows, X_t_cols)]
    [X_t_rows, X_t_cols] = np.where(X_t > 0.5)
    X_t_values = X_t[(X_t_rows, X_t_cols)]

    N = len(X_t_values)  # number of sample point
    para = np.array([V, K, J, N], dtype=np.int32)

    X_t_values = np.array(X_t_values, dtype=np.int32)
    X_t_rows = np.array(X_t_rows, dtype=np.int32)
    X_t_cols = np.array(X_t_cols, dtype=np.int32)

    Xt_to_t1_t = np.zeros([K, J], dtype=np.float32, order='C')
    WSZS_t = np.zeros([V, K], dtype=np.float32, order='C')
    Phi_t = np.array(Phi_t, dtype=np.float32, order='C')
    Theta_t = np.array(Theta_t, dtype=np.float32, order='C')

    randomseed=np.random.rand(N)
    block_x = int(400)
    grid_x = int(np.floor(N / block_x) + 1)
    randomseed=np.array(randomseed, dtype=np.float32, order='C')
    func(drv.In(randomseed), drv.In(para), drv.In(X_t_values), drv.In(X_t_rows), drv.In(X_t_cols), drv.In(Phi_t),
         drv.In(Theta_t), drv.InOut(WSZS_t), drv.InOut(Xt_to_t1_t),
         grid=(grid_x, 1, 1), block=(block_x, 1, 1))

    # Multi_lib.Multi_Sample(X_t, Phi_t, Theta_t, WSZS_t, Xt_to_t1_t, V, K, J)
    #    V = X_t.shape[0]
    #    J = X_t.shape[1]
    #    K = Theta_t.shape[0]
    #    a = Sampler()
    #    Xt_to_t1_t = np.zeros([K,J])
    #    WSZS_t = np.zeros([V,K])
    #    Embedding = np.zeros(K,dtype='uint32')
    #    Augment_Matrix = np.zeros([V,K,J],dtype='uint32')
    #
    #    for n in range(Xt_xy.shape[1]):
    #        v = Xt_xy[0,n]
    #        j = Xt_xy[1,n]
    #        Pro =  (Phi_t[v,:] *Theta_t[:,j])/(Phi_t[v,:] *Theta_t[:,j]).sum(0)
    #        a.multinomial(X_t[v,j],Pro,Embedding)   ## output to embedding
    #        Augment_Matrix[v,:,j] = Embedding
    #    Xt_to_t1_t[:,:] = Augment_Matrix.sum(0);
    #    WSZS_t[:,:] = Augment_Matrix.sum(2);
    return Xt_to_t1_t, WSZS_t

def Multrnd_Input(X_t,Phi_1,Theta_1,Phi_2,Theta_2,k_1,k_2):
    V = X_t.shape[0]
    J = X_t.shape[1]
#    Pro1 = np.dot(Phi_1,Theta_1).astype('double') / (np.dot(Phi_1,Theta_1).astype('double').sum(0) + 1)
#    if np.dot(Phi_1,Theta_1).sum(0) == 0:
#        Pro1 = np.ones([V,J]).astype('double') 
#    else:
#    Pro1 = np.dot(Phi_1,Theta_1).astype('double') / np.maximum( np.dot(Phi_1,Theta_1).sum(0),1 ) 
  
    Pro1 = np.dot(Phi_1,Theta_1).astype('double') / np.maximum( (np.sqrt( np.dot(Phi_1,Theta_1) * np.dot(Phi_1,Theta_1) ).sum(0)),1 ) 
    Pro1 = Pro1 * k_1
#    Pro2 = np.dot(Phi_2,Theta_2).astype('double') / (np.dot(Phi_2,Theta_2).astype('double').sum(0) + 1)
#    if np.dot(Phi_2,Theta_2).sum(0) == 0:
#        Pro2 = np.ones([V,J]).astype('double') 
#    else:            
    Pro2 = np.dot(Phi_2,Theta_2).astype('double') / np.maximum( (np.sqrt( np.dot(Phi_2,Theta_2) * np.dot(Phi_2,Theta_2) ).sum(0)),1 )
#    Pro2 = np.dot(Phi_2,Theta_2).astype('double') / np.maximum( np.dot(Phi_2,Theta_2).sum(0),1 )
    Pro2 = Pro2 * k_2
    X_t_1 = np.zeros([V,J], order = 'C').astype('double')
    X_t_2 = np.zeros([V,J], order = 'C').astype('double')
    
    Multi_lib.Multi_Input(X_t.astype('double'),Pro1,Pro2,X_t_1,X_t_2,V,J)
    return X_t_1,X_t_2
    
def Crt_Multirnd_Matrix(Xt_to_t1_t,Phi_t1,Theta_t1):
    Kt = Xt_to_t1_t.shape[0]
    J = Xt_to_t1_t.shape[1]
    Kt1 = Theta_t1.shape[0]
    Xt_to_t1_t1 = np.zeros([Kt1,J],order = 'C').astype('double')
    WSZS_t1 = np.zeros([Kt,Kt1],order = 'C').astype('double')
    
    Crt_Multi_lib.Crt_Multi_Sample(Xt_to_t1_t, Phi_t1,Theta_t1, WSZS_t1, Xt_to_t1_t1, Kt, Kt1 , J)
    
    
#    Kt = Xt_to_t1_t.shape[0]
#    J = Xt_to_t1_t.shape[1]
#    Kt1 = Theta_t1.shape[0]
#    Xt1 = np.zeros([Kt,J])
#    a = Sampler()
#    Xt_to_t1_t1 = np.zeros([Kt1,J])
#    WSZS_t1 = np.zeros([Kt,Kt1])
#    Embedding = np.zeros(Kt1,dtype='uint32')
#    Augment_Matrix = np.zeros([Kt,Kt1,J],dtype='uint32')
#    for k in range(Kt):
#        for j in range(J):
#            if Xt_to_t1_t[k,j] < 0.5:
#                continue
#            else:
#                Xt1[k,j] = a.crt(Xt_to_t1_t[k,j], np.dot(Phi_t1[k,:],Theta_t1[:,j]))
#                Pro = Phi_t1[k,:]*Theta_t1[:,j] / (Phi_t1[k,:]*Theta_t1[:,j]).sum(0)
#                a.multinomial(Xt1[k,j],Pro,Embedding)
#                Augment_Matrix[k,:,j] = Embedding
#    Xt_to_t1_t1[:,:] = Augment_Matrix.sum(0)
#    WSZS_t1[:,:] = Augment_Matrix.sum(2)
    return Xt_to_t1_t1 , WSZS_t1

def Crt_Matrix(Xt_to_t1_t, p ):
    Kt = Xt_to_t1_t.shape[0]
    J = Xt_to_t1_t.shape[1]
    Xt_t1 = np.zeros([Kt,J],order = 'C').astype('double')
    
    Crt_lib.Crt_Sample(Xt_to_t1_t, p.astype('double'), Xt_t1, Kt, J)
    
    
#    Kt = Xt_to_t1_t.shape[0]
#    J = Xt_to_t1_t.shape[1]
#    Kt1 = Theta_t1.shape[0]
#    Xt1 = np.zeros([Kt,J])
#    a = Sampler()
#    Xt_to_t1_t1 = np.zeros([Kt1,J])
#    WSZS_t1 = np.zeros([Kt,Kt1])
#    Embedding = np.zeros(Kt1,dtype='uint32')
#    Augment_Matrix = np.zeros([Kt,Kt1,J],dtype='uint32')
#    for k in range(Kt):
#        for j in range(J):
#            if Xt_to_t1_t[k,j] < 0.5:
#                continue
#            else:
#                Xt1[k,j] = a.crt(Xt_to_t1_t[k,j], np.dot(Phi_t1[k,:],Theta_t1[:,j]))
#                Pro = Phi_t1[k,:]*Theta_t1[:,j] / (Phi_t1[k,:]*Theta_t1[:,j]).sum(0)
#                a.multinomial(Xt1[k,j],Pro,Embedding)
#                Augment_Matrix[k,:,j] = Embedding
#    Xt_to_t1_t1[:,:] = Augment_Matrix.sum(0)
#    WSZS_t1[:,:] = Augment_Matrix.sum(2)
    return Xt_t1

    
def Sample_Phi(WSZS_t,Eta_t):   # (array, scalar)
    Kt = WSZS_t.shape[0]
    Kt1 = WSZS_t.shape[1]
    Phi_t_shape = WSZS_t + Eta_t
    Phi_t = np.zeros([Kt,Kt1])
    Phi_t = np.random.gamma(Phi_t_shape,1)
#    for kt in range(Kt):
#        for kt1 in range(Kt1):
#            Phi_t[kt,kt1] = a.gamma(Phi_t_shape[kt,kt1],1) 
    Phi_t = Phi_t / Phi_t.sum(0)
    return Phi_t
    
def Sample_Theta(Xt_to_t1_t,c_j_t1,p_j_t,shape):
    Kt = Xt_to_t1_t.shape[0]
    N = Xt_to_t1_t.shape[1]
    Theta_t = np.zeros([Kt,N])
    Theta_t_shape = Xt_to_t1_t + shape
    Theta_t[:,:] = np.random.gamma(Theta_t_shape,1) / (c_j_t1[0,:]-np.log(np.maximum(realmin,1-p_j_t[0,:])))
#    a = Sampler()
#    for kt in range(Kt):
#        for n in range(N):
#            Theta_t[kt,n] = a.gamma(Theta_t_shape[kt,n],1)/(c_j_t1[0,n]-np.log(np.maximum(realmin,1-p_j_t[0,n])))
    return Theta_t


def Sample_Theta_2(Xt_to_t1_t,shape):
    Kt = Xt_to_t1_t.shape[0]
    N = Xt_to_t1_t.shape[1]
    Theta_t = np.zeros([Kt,N])
    Theta_t_shape = Xt_to_t1_t + shape
    Theta_t[:,:] = np.random.gamma(Theta_t_shape,1) 
#    a = Sampler()
#    for kt in range(Kt):
#        for n in range(N):
#            Theta_t[kt,n] = a.gamma(Theta_t_shape[kt,n],1)/(c_j_t1[0,n]-np.log(np.maximum(realmin,1-p_j_t[0,n])))
    return Theta_t

def ProjSimplexSpecial(Phi_tmp,Phi_old,epsilon):
    Phinew = Phi_tmp - (Phi_tmp.sum(0) - 1) * Phi_old
    if  np.where(Phinew[:,:]<=0)[0].size >0:
        Phinew = np.maximum(epsilon,Phinew)
        Phinew = Phinew/np.maximum(realmin,Phinew.sum(0))
    return Phinew

def Reconstruct_error(X,Phi,Theta):
    return np.power(X-np.dot(Phi,Theta),2).sum()
    
    
