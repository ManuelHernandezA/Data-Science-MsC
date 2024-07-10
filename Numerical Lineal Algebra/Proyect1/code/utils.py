import numpy as np

def Newton_step(lamb0,dlamb,s0,ds):
    alp=1
    idx_lamb0=np.array(np.where(dlamb<0))
    if idx_lamb0.size>0:
        alp = min(alp,np.min(-lamb0[idx_lamb0]/dlamb[idx_lamb0]))
    idx_s0=np.array(np.where(ds<0))
    if idx_s0.size>0:
        alp = min(alp,np.min(-s0[idx_s0]/ds[idx_s0]))
    return alp


def evalF_noA(G,x,g,C,lamb,s,d):
    rL = np.dot(G,x) + g - np.dot(C, lamb)
    rC = s + d - np.dot(C.T, x)
    rS = s * lamb
    return np.concatenate((rL, rC, rS))


def open_matrix(path, n, m):
    mtx = np.zeros((n,m))
    with open(path) as file:
        for x in file.readlines():
            tmp = np.array(x.split()).astype(float)
            mtx[tmp[0].astype(int)-1, tmp[1].astype(int)-1] = tmp[2]
    return mtx


def open_vector(path, n):
    vector = np.zeros(n)
    with open(path) as file:
        for x in file.readlines():
            tmp = np.array(x.split()).astype(float)
            vector[tmp[0].astype(int)-1] = tmp[1]
    return vector


def evalF(G, x, g, A, gamma, C, lamb, b, s, d):
    rL = np.dot(G,x) + g - np.dot(A, gamma) - np.dot(C, lamb)
    rA = b - np.dot(A.T,x)
    rC = s + d - np.dot(C.T, x)
    rS = np.multiply(s,lamb)
    return np.concatenate((rL, rA, rC, rS))