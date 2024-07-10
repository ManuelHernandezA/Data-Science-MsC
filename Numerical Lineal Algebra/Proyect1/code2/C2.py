import numpy as np
from utils import evalF_noA, Newton_step

def C2_test(n, get_cond=False):
    dc = {"condition_numbers": [], "iterations": 0, "precision": 0}

    epsilon = 10e-16
    m = 2*n
    G = np.identity(n)
    C = np.column_stack((np.identity(n), -np.identity(n)))
    d = np.array([-10]*m)
    g = np.random.normal(0,1,n)

    x = np.zeros(n)
    s = np.ones(m)
    lamb = np.ones(m)
    z = np.concatenate((x,lamb,s))
    while True:
        M_kkt = np.row_stack((np.column_stack((G,                -C,                 np.zeros((n, m)))),
                            np.column_stack((-C.T,             np.zeros((m,m)),    np.identity(m))),
                            np.column_stack((np.zeros((m, n)), np.diag(s),         np.diag(lamb)))
        ))
        # Step 1 solve Newton step
        F_Z = evalF_noA(G,x,g,C,lamb,s,d)

        if np.linalg.norm(F_Z[:n]) < epsilon:
            break
        if np.linalg.norm(F_Z[n:n+m]) < epsilon:
            break

        d_z = np.linalg.solve(M_kkt, -F_Z)
        # Step 2 step size correction substep
        dlamb = d_z[n:n+m]
        ds = d_z[n+m:]
        alpha = Newton_step(lamb,dlamb,s,ds)
        # Step 3 compute sigma and mu
        mu = np.dot(s.T, lamb)/m

        if mu < epsilon:
            break

        mu_tilde = np.dot((s + alpha * ds).T, (lamb + alpha*dlamb))/m
        sigma = (mu_tilde/mu)**3
        # Step 4 corrector substep
        F_Z = np.concatenate((-F_Z[:n+m],-F_Z[n+m:]-ds*dlamb + np.full(m, sigma*mu)))
        d_z = np.linalg.solve(M_kkt, F_Z)
        # Step 5 step size correction substep
        dlamb = d_z[n:n+m]
        ds = d_z[n+m:]
        alpha = Newton_step(lamb,dlamb,s,ds)
        # Step 6 update substep
        z = z + 0.95*alpha*d_z
        x = z[:n]
        lamb = z[n:n+m]
        s = z[n+m:]
        dc["iterations"] += 1
        if get_cond: dc["condition_numbers"].append(np.linalg.cond(M_kkt))
        if dc["iterations"] > 100:
            break
    dc["precision"] = np.linalg.norm(z[:n] + g)/np.linalg.norm(g)
    return z, dc

print("Calculating C2 for N in range [10, 500] with step 10")
for i in range(10, 501, 10):
    z, dc= C2_test(i)
    print(f"\tFor N = {i}, done in {dc['iterations']} iterations ")