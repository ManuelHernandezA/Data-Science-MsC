from scipy.io import mmread

A = mmread('p2p-Gnutella30.mtx')

import numpy as np
from scipy.sparse import diags

def compute_PR(G, m=0.15, tol=1e-8):
    n = G.shape[0]
    x_k = np.ones(n)/n
    x_k1 = None
    n_j = G.sum(axis=0)
    D = np.array([0 if x == 0 else 1/x for x in np.array(n_j)[0]])
    z = np.array([1/n if x == 0 else m/n for x in D])
    e = np.ones(n)
    e_z = e@z
    D = diags(D)
    k = 0
    A = G@D
    while True:
        k+=1
        x_k1 = ((1-m)*A)@x_k.T + e_z*x_k.T
        if np.linalg.norm(x_k1-x_k,ord=np.inf) < tol:
            break
        else:
            x_k = x_k1.copy()
    return x_k1, k

print(compute_PR(A))


def compute_PR2(G, m=0.15, tol=1e-8):
    n = G.shape[0]
    x = np.ones(n)/n
    L = [set() for _ in range(n)]
    for i, j in zip(*G.nonzero()): L[j].add(i) 
    n_j = [len(j) for j in L]
    k = 0

    while True:
        k+=1
        xc=x
        x=np.zeros(n)
        for j in range (0,n):
            if(n_j[j]==0):
                x=x+xc[j]/n
            else:
                for i in L[j]:
                    x[i]=x[i]+xc[j]/n_j[j]
        x=(1-m)*x+m/n
        if np.linalg.norm(x-xc,ord=np.inf) < tol:
            break
    return x, k

print(compute_PR2(A))