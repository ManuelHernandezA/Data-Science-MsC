import numpy as np
from utils import evalF, Newton_step, open_matrix, open_vector
from timeit import default_timer as timer
from scipy.linalg import lapack

def C6_general(A, b, C, d, G, g):
    epsilon = 10e-16
    m = d.shape[0]
    n = A.shape[0]
    p = A.shape[1]
    
    x = np.zeros(n)
    gamma = np.ones(p)
    lamb = np.ones(m)
    s = np.ones(m)
    
    z = np.concatenate((x,gamma,lamb,s))
    k = 0
    start = timer()
    while True:
        lamb_inv = np.diag(1./lamb)
        M_kkt = np.row_stack((np.column_stack((G,                   -A,                 -C)),
                            np.column_stack((-A.T,                  np.zeros((p,p)),    np.zeros((p,m)))),
                            np.column_stack((-C.T,                  np.zeros((m,p)),    -np.dot(lamb_inv,np.diag(s))))
        ))
        # Step 1 solve Newton step
        F_Z = evalF(G, x, g, A, gamma, C, lamb, b, s, d)

        r1 = F_Z[:n]            # rL
        r2 = F_Z[n:n+p]         # rA
        r3 = F_Z[n+p:n+p+m]     # rC
        r4 = F_Z[n+p+m:]        # rs
        
        if np.linalg.norm(r1) < epsilon:
            break
        if np.linalg.norm(r2) < epsilon:
            break
        if np.linalg.norm(r3) < epsilon:
            break
        
        r_vector = np.concatenate((-r1, -r2, -r3 + np.dot(lamb_inv, r4)))

        delta = lapack.dsysv(M_kkt, r_vector)[2]
        # Step 2 step size correction substep
        dlamb = delta[n+p:n+p+m]
        ds = np.dot(lamb_inv, -r4 - np.dot(np.diag(s),dlamb))
        alpha = Newton_step(lamb,dlamb,s,ds)
        # Step 3 compute sigma and mu
        mu = np.dot(s.T, lamb)/m

        if abs(mu) < epsilon:
            break

        mu_tilde = np.dot((s + alpha * ds).T, (lamb + alpha*dlamb))/m
        sigma = np.power(mu_tilde/mu, 3)
        # Step 4 corrector substep
        r4 = r4 + ds*dlamb - np.full(m, sigma*mu)
        r_vector = -np.concatenate((r1,r2,r3 - np.dot(lamb_inv, r4)))
        
        delta = lapack.dsysv(M_kkt, r_vector)[2]
        # Step 5 step size correction substep
        dlamb = delta[n+p:n+p+m]
        ds = np.dot(lamb_inv, - r4 - s*dlamb)

        alpha = Newton_step(lamb,dlamb,s,ds)
        # Step 6 update substep
        z = z + 0.95*alpha*np.concatenate((delta, ds))
        x = z[:n]
        gamma = z[n:n+p]
        lamb = z[n+p:n+p+m]
        s = z[n+p+m:]
        k+=1
        if k > 100:
            break
    end = timer()
    print(f"Elapsed time for computation: {end - start:0.4f}s")
    return z

n = 100
p = 50
m = 200

A = open_matrix("./optpr1/A.dad", n, p)
b = open_vector("./optpr1/b.dad", p)
C = open_matrix("./optpr1/C.dad", n, m)
d = open_vector("./optpr1/d.dad", m)
G = open_matrix("./optpr1/G.dad", n, n)
g = open_vector("./optpr1/g_vector.dad", n)

G = G + G.T -np.diag(np.diag(G))

print("Computing C5 for optpr1...")
z = C6_general(A, b, C, d, G, g)
evalx = z[:n]
print("\tf(x) =",(1/2)*np.dot(evalx.T, np.dot(G, evalx)) + np.dot(g.T, evalx))

n = 1000
p = 500
m = 2000

A = open_matrix("./optpr2/A.dad", n, p)
b = open_vector("./optpr2/b.dad", p)
C = open_matrix("./optpr2/C.dad", n, m)
d = open_vector("./optpr2/d.dad", m)
G = open_matrix("./optpr2/G.dad", n, n)
g = open_vector("./optpr2/g_vector.dad", n)

G = G + G.T -np.diag(np.diag(G))

print("Computing C5 for optpr2...")
z = C6_general(A, b, C, d, G, g)
evalx = z[:n]
print("\tf(x) =",(1/2)*np.dot(evalx.T, np.dot(G, evalx)) + np.dot(g.T, evalx))