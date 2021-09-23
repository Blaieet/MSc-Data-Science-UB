import numpy as np
import sys
import time
from scipy.linalg import solve_triangular, cholesky

np.random.seed(1997)

def createMatrix(lamb,s):
    # Create M_KKT, S and Lambda matrices
    Lamb = np.diag(lamb)
    S = np.diag(s)
    M = G + C.dot(np.diag(1/s*lamb)).dot(np.transpose(C))
    return M,Lamb,S


# Original function, f(x)
def f(x):
    return 0.5*np.transpose(x).dot(G).dot(x)+np.transpose(g).dot(x)


# Function F(z)
def fun(x,lamb,s):
    comp1 = G.dot(x)+g-C.dot(lamb)
    comp2 = s+d-np.transpose(C).dot(x)
    comp3 = s*lamb
    return np.concatenate((comp1,comp2,comp3))


# Function that implements the step-size substep
def Newton_step(lamb0,dlamb,s0,ds):
    alp=1
    idx_lamb0=np.array(np.where(dlamb<0))
    if idx_lamb0.size>0:
        alp = min(alp,np.min(-lamb0[idx_lamb0]/dlamb[idx_lamb0]))
    idx_s0=np.array(np.where(ds<0))
    if idx_s0.size>0:
        alp = min(alp,np.min(-s0[idx_s0]/ds[idx_s0]))
    return alp


# Solves the problem F(z)=0 using a variant of the Newton's method
def Newton(maxIter=100,epsilon=10e-16):
    global lamb,s,x,z
    start_time = time.time()

    Ghat,Lamb,S = createMatrix(lamb,s)


    for i in range(maxIter):
        S_inv = np.diag(1/s)

        # Standard Newton step (solve KKT system, Strategy 2)...
        b = fun(x,lamb,s)
        r1, r2, r3 = b[:n], b[n:n+m], b[n+m:]
        rhat = -C.dot(np.diag(1/s)).dot((-r3+lamb*r2))
        b = -r1-rhat

        # ...using Cholesky factorization
        L = cholesky(Ghat,lower=True)

        y = solve_triangular(L,b,lower=True)
        delta_x = solve_triangular(np.transpose(L),y)
        delta_lamb = S_inv.dot((-r3+lamb*r2))-S_inv.dot(Lamb.dot(np.transpose(C)).dot(delta_x))
        delta_s = -r2 + np.transpose(C).dot(delta_x)
        delta = np.concatenate((delta_x,delta_lamb, delta_s))

        # Step-size correction substep
        alpha = Newton_step(lamb,delta[n:n+m],s,delta[n+m:])

        # Compute correction parameters
        mu = s.dot(lamb)/m
        muTilde = ((s+alpha*delta[n+m:]).dot(lamb+alpha*delta[n:n+m]))/m
        sigma = (muTilde/mu)**3

        # Corrector substep...
        Ds_Dlamb = np.diag(delta[n+m:]*delta[n:n+m])
        b = -r1-(-C.dot(np.diag(1/s)).dot((-r3-Ds_Dlamb.dot(e)+sigma*mu*e+lamb*r2)))

        # ...using Cholesky factorization again
        y = solve_triangular(L,b,lower=True)
        delta_x = solve_triangular(np.transpose(L),y)
        delta_lamb = S_inv.dot(-r3-Ds_Dlamb.dot(e)+sigma*mu*e+lamb*r2)-S_inv.dot(lamb*(np.transpose(C).dot(delta_x)))
        delta_s = -r2 + np.transpose(C).dot(delta_x)
        delta = np.concatenate((delta_x,delta_lamb, delta_s))

        # Step-size correction substep
        alpha = Newton_step(lamb,delta[n:n+m],s,delta[n+m:])

        # Update substep
        z = z+0.95*alpha*delta

        # Stop criterion
        if (np.linalg.norm(-r1) < epsilon) or (np.linalg.norm(-r2) < epsilon) or (np.abs(mu) < epsilon):
            break

        # Update M_KKT
        x = z[:n]
        lamb = z[n:n+m]
        s = z[n+m:]
        Ghat,Lamb,S = createMatrix(lamb,s)

    end_time = time.time()

    print('\nProblem dimension:',n)
    print('Found minimum:',f(x))
    print('Actual minimum:',f(-g))
    print('Iterations:',i)
    print("Computation time (sec): ",end_time-start_time)


if __name__ == '__main__':

    # Read dimension (3 by default)
    if len(sys.argv) == 1:
        print("You can change the dimension of the problem using 'python Newton.py [n]'")
        n = 3
    else:
        n = int(sys.argv[1])

    # Problem dimensions
    m = 2*n

    # Problem data
    G = np.identity(n)
    C = np.concatenate((G,-G),axis=1)
    d = np.full((m),-10)
    e = np.ones((m))
    g = np.random.normal(0,1,(n))
    x = np.zeros((n))
    lamb = np.ones((m))
    s = np.ones((m))
    z = np.concatenate((x,lamb,s))

    # Call algorithm
    Newton()
