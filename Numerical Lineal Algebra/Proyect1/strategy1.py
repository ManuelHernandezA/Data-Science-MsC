def get_r(G, C, g, d, x,lamb,s):
    r1 = G.dot(x)+g-C.dot(lamb)
    r2 = s+d-np.transpose(C).dot(x)
    r3 = s*lamb
    return np.concatenate((r1,r2,r3))

# Define the functions
def get_mkk(G, C, x, lamb, s):
    
    S = np.diag(s)
    A = np.diag(lamb)
    A_inv = np.diag(1 / lamb)
    
    M_kkt = np.concatenate(
        (np.concatenate((G, -C), axis=1),
         np.concatenate((-C.T, -np.dot(A_inv,S)), axis=1))
    )

    return M_kkt

def get_r_mkk(r1, r2, r3, A_inv):
    return np.concatenate((r1,
                           r2 - np.dot(A_inv, r3)))

def get_delta(M_kkt, r):
    l, d, p = scipy.linalg.ldl(M_kkt)
    sub_delta = scipy.linalg.solve_triangular(l, r, unit_diagonal=True, lower=True)
    delta = scipy.linalg.solve_triangular(np.dot(d,l.T), sub_delta)
    return delta

# Execution of the program
times=[]
iterations=[]
precision_error=[]
for n in range(2, 200):
    #define vectors
    m = 2 * n
    x = np.zeros(n)
    lamb = np.ones(m)
    s = np.ones(m)
    d = np.ones(m) * (-10)
    g = np.random.random(n)
    
    #define matrices
    G = np.identity(n)
    C = np.zeros((n, m))
    for i in range(0, n):
        for j in range(0, m):
            if i == j:
                C[i, j] = 1
            elif i == j % n:
                C[i, j] = -1

    start_time = time.time()
    for i in range(0, 100):
        M_kkt = get_mkk(G, C, x, lamb, s)
        
        A_inv = np.diag(1 / lamb)
               
        r = get_r(G, C, g, d, x, lamb, s)
        
        if (np.linalg.norm(r[:n]) < ERROR) or (np.linalg.norm(r[n:n+m]) < ERROR) or (np.linalg.norm(r[n+m:]) < ERROR):
            break
        
        r_mkk = get_r_mkk(r[:n], r[n:n+m], r[n+m:], A_inv)

        # 1. Predictor substep
        delta = get_delta(M_kkt, -r_mkk)
        delta_x = delta[:n]
        delta_lambda = delta[n:]
        delta_s = np.dot(A_inv, -r[n+m:] - np.multiply(s, delta_lambda))
        # 2. Step-size correction substep
        step = Newton_step(lamb, delta_lambda, s, delta_s)

        # 3.
        mu = (np.dot(s.T, lamb)) / m

        if abs(mu) < ERROR:
            break

        mu_est = np.dot((s + step * delta_s).T, (lamb + step * delta_lambda)) / m
        step = np.power(mu_est / mu, 3)
        
        # 4. Corrector substep
        r[n+m:] = r[n+m:] + np.multiply(d,s) - step*mu*np.ones(m)
        r_est = get_r_mkk(r[:n], r[n:n+m], r[n+m:], A_inv)

        delta = get_delta(M_kkt, -r_est)
        delta_x = delta[:n]
        delta_lambda = delta[n:]             
        delta_s = np.dot(A_inv, -r[n+m:] - np.multiply(s, delta_lambda))

        # 5. Step-size correction substep
        step = Newton_step(lamb, delta_lambda, s, delta_s)

        # 6. Update substep
        x = x + 0.95 * step * delta[:n]
        lamb = lamb + 0.95 * step * delta[n:]
        s = s + 0.95 * step * delta_s
        
    end_time = time.time()
    times.append(end_time - start_time)
    iterations.append(i+1)
    precision_error.append(np.linalg.norm(-g-x))