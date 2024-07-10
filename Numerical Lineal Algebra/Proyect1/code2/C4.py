import numpy as np
from utils import evalF_noA, Newton_step
from timeit import default_timer as timer
from scipy.linalg import solve_triangular, ldl

def C4_strategy_1(n, get_cond=False):
    dc = {"condition_numbers" : [], "iterations" : 0, "precision" : 0, "time" : 0}
    
    epsilon = 10e-16
    m = 2*n
    G = np.identity(n)
    C = np.column_stack((np.identity(n), -np.identity(n)))
    d = np.array([-10.]*m)
    g = np.random.normal(0.,1.,n)

    x = np.zeros(n)
    s = np.ones(m)
    lamb = np.ones(m)
    z0 = np.concatenate((x,lamb,s))
    
    start = timer()
    while True:
        M_kkt = np.row_stack((np.column_stack((G,                -C)),
                            np.column_stack((-C.T,              -np.dot(np.diag(1./lamb),np.diag(s))))
        ))
        
        lamb_inv = np.diag(1./lamb)
        # Step 1 solve Newton step
        F_Z = evalF_noA(G,x,g,C,lamb,s,d)

        r1 = F_Z[:n]
        r2 = F_Z[n:n+m]
        r3 = F_Z[n+m:]

        if np.linalg.norm(r1) < epsilon:
            break
        if np.linalg.norm(r2) < epsilon:
            break

        r_vector = -np.concatenate((r1, r2-np.dot(lamb_inv, r3)))

        L, D, _ = ldl(M_kkt)
        y = solve_triangular(L, r_vector, unit_diagonal=True, lower=True)
        d_z = solve_triangular(np.dot(D,L.T), y)

        # Step 2 step size correction substep
        dlamb = d_z[n:]
        ds = np.dot(lamb_inv,-r3-s*dlamb)
        alpha = Newton_step(lamb,dlamb,s,ds)
        # Step 3 compute sigma and mu
        mu = np.dot(s.T, lamb)/m

        if abs(mu) < epsilon:
            break

        mu_tilde = np.dot((s + alpha * ds).T, (lamb + alpha*dlamb))/m
        sigma = (mu_tilde/mu)**3

        # Step 4 corrector substep
        r3 = r3 + ds*dlamb - np.full(m, sigma*mu)
        r_vector = -np.concatenate((r1, 
                                    r2-np.dot(lamb_inv, r3)))
        
        y = solve_triangular(L, r_vector, unit_diagonal=True, lower=True)
        d_z = solve_triangular(np.dot(D,L.T), y)

        # Step 5 step size correction substep
        dlamb = d_z[n:]
        ds = np.dot(lamb_inv,-r3-s*dlamb)
        alpha = Newton_step(lamb,dlamb,s,ds)
        # Step 6 update substep
        z0 = z0 + 0.95*alpha*np.concatenate((d_z, ds))
        x = z0[:n]
        lamb = z0[n:n+m]
        s = z0[n+m:]
        
        dc["iterations"] += 1
        if get_cond: dc["condition_numbers"].append(np.linalg.cond(M_kkt))
        if dc["iterations"] > 100:
            break
    end = timer()
    dc["precision"] = np.linalg.norm(z0[:n] + g)/np.linalg.norm(g)
    dc["time"] = end - start
    return z0, dc


def C4_strategy_2(n, get_cond=False):
    dc = {"condition_numbers" : [], "iterations" : 0, "precision" : 0, "time" : 0}
    
    epsilon = 10e-16
    m = 2*n
    G = np.identity(n)
    C = np.column_stack((np.identity(n), -np.identity(n)))
    d = np.array([-10.]*m)
    g = np.random.normal(0.,1.,n)

    x = np.zeros(n)
    s = np.ones(m)
    lamb = np.ones(m)
    z0 = np.concatenate((x,lamb,s))

    start = timer()
    while True:
        s_inv =np.diag(1./s)
        M_kkt = G+np.dot(C,np.dot(s_inv,np.dot(np.diag(lamb),C.T)))
        
        # Step 1 solve Newton step
        F_Z = evalF_noA(G,x,g,C,lamb,s,d)

        r1 = F_Z[:n]
        r2 = F_Z[n:n+m]
        r3 = F_Z[n+m:]

        if np.linalg.norm(r1) < epsilon:
            break
        if np.linalg.norm(r2) < epsilon:
            break

        r_vector = -r1+np.dot(C,np.dot(s_inv, -r3 + np.dot(np.diag(lamb),r2)))

        L = np.linalg.cholesky(M_kkt)
        y = solve_triangular(L, r_vector, lower=True)
        dx = solve_triangular(L.T, y)
        
        # Step 2 step size correction substep
        dlamb = np.dot(s_inv, -r3+np.dot(np.diag(lamb), r2)) - np.dot(s_inv,np.dot(np.diag(lamb), np.dot(C.T, dx)))
        ds = -r2+np.dot(C.T,dx)
        alpha = Newton_step(lamb,dlamb,s,ds)
        # Step 3 compute sigma and mu
        mu = np.dot(s.T, lamb)/m

        if abs(mu) < epsilon:
            break

        mu_tilde = np.dot((s + alpha * ds).T, (lamb + alpha*dlamb))/m
        sigma = (mu_tilde/mu)**3

        # Step 4 corrector substep
        r3 = r3 + ds*dlamb - np.full(m, sigma*mu)
        r_vector = -r1+np.dot(C,np.dot(s_inv, -r3 - ds*dlamb + np.full(m, sigma*mu)))
        
        y = solve_triangular(L, r_vector, lower=True)
        dx = solve_triangular(L.T, y)

        # Step 5 step size correction substep
        dlamb = np.dot(s_inv, -r3+np.dot(np.diag(lamb), r2)) - np.dot(s_inv,np.dot(np.diag(lamb), np.dot(C.T, dx)))
        ds = -r2+np.dot(C.T,dx)
        alpha = Newton_step(lamb,dlamb,s,ds)
        # Step 6 update substep
        z0 = z0 + 0.95*alpha*np.concatenate((dx, dlamb, ds))
        x = z0[:n]
        lamb = z0[n:n+m]
        s = z0[n+m:]
        
        dc["iterations"] += 1
        if get_cond: dc["condition_numbers"].append(np.linalg.cond(M_kkt))
        if dc["iterations"] > 100:
            break
    end = timer()
    dc["precision"] = np.linalg.norm(z0[:n] + g)/np.linalg.norm(g)
    dc["time"] = end - start
    return z0, dc

print("Calculating C4 with strategy 1 for N in range [10, 500] with step 10")
for i in range(10, 501, 10):
    z, dc= C4_strategy_1(i)
    print(f"\tFor N = {i}, done in {dc['time']:0.4f} seconds with {dc['iterations']} iterations")

print("Calculating C4 with strategy 2 for N in range [10, 500] with step 10")
for i in range(10, 501, 10):
    z, dc= C4_strategy_2(i)
    print(f"\tFor N = {i}, done in {dc['time']:0.4f} seconds with {dc['iterations']} iterations")
