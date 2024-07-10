import numpy as np
import scipy.linalg as lng
from scipy.linalg import lstsq

def solve_LSP_QR(A, b):
    n = A.shape[1]
    Q, R, P = lng.qr(A, pivoting=True)
    rank = np.linalg.matrix_rank(A)

    R1 = R[:rank, :rank]

    C_D = Q.T @ b
    C = C_D[:rank]

    u = lng.solve_triangular(R1, C)
    v = np.zeros(n-rank) # set v to zero

    P = np.eye(n)[:,P] # permutation matrix
    x = np.linalg.solve(P.T, np.concatenate((u, v)))

    return x

def solve_LSP_SVD(A, b):
    u, s, v = np.linalg.svd(A)
    for i in range(s.shape[0]):
        if s[i] < 1e-10:
            s[i] = 0
        else:
            s[i] = 1/s[i]
    s_plus = np.concatenate((np.diag(s), np.zeros((s.shape[0], u.shape[0]-s.shape[0]))), axis=1)
    Ainv = np.dot(v.T, np.dot(s_plus, u.T))
    return Ainv@b

## DATASET 1: datafile.csv

data = np.genfromtxt("datafile.csv")
x_vec = data[:,0]
A = np.zeros((x_vec.shape[0], 6))
for m in range(6):
    A[:,m] = x_vec**m
b = data[:,1]


print("LSP solutions for dataset 1:")
x = solve_LSP_QR(A, b)
print("\tNorm of QR solution:", np.linalg.norm(A@x-b))

x = solve_LSP_SVD(A, b)
print("\tNorm of SVD solution:", np.linalg.norm(A@x-b))

print("\tNorm of Scipy's solution:", np.linalg.norm(A@lstsq(A,b)[0]-b))

## DATASET 2: datafile2.csv

data = np.genfromtxt("datafile2.csv", delimiter=",")
A = data[:,:11]
b = data[:,11]

print("LSP solutions for dataset 2:")
x = solve_LSP_QR(A, b)
print("\tNorm of QR solution:", np.linalg.norm(A@x-b))

x = solve_LSP_SVD(A, b)
print("\tNorm of SVD solution:", np.linalg.norm(A@x-b))

print("\tNorm of Scipy's solution:", np.linalg.norm(A@lstsq(A,b)[0]-b))