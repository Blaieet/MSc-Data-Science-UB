import numpy as np
import sys
import time

np.random.seed(1997)

# Creates the M_KKT, Lambda and S matrices
def createMatrix(lamb,s):
    Lamb = np.diag(lamb)
    S = np.diag(s)
    row1 = np.concatenate((G,-C,np.zeros((n,m))),axis=1)
    row2 = np.concatenate((np.transpose(-C),np.zeros((m,m)),np.identity(m)),axis=1)
    row3 = np.concatenate((np.zeros((m,n)),S,Lamb),axis=1)
    M = np.concatenate((row1,row2,row3))
    return M,S,Lamb


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

    # Create M_KKT, S and Lambda matrices
    M,S,Lamb = createMatrix(lamb,s)

    for i in range(maxIter):

        # Standard Newton step (solve KKT system)
        b = -fun(x,lamb,s)
        delta = np.linalg.solve(M,b)

        # Step-size correction substep
        alpha = Newton_step(lamb,delta[n:n+m],s,delta[n+m:])

        # Compute correction parameters
        mu = s.dot(lamb)/m
        muTilde = ((s+alpha*delta[n+m:]).dot(lamb+alpha*delta[n:n+m]))/m
        sigma = (muTilde/mu)**3

        # Corrector substep
        b[n+m:] = b[n+m:]-np.diag(delta[n+m:]*delta[n:n+m]).dot(e)+sigma*mu*e
        delta = np.linalg.solve(M,b)

        # Step-size correction substep
        alpha = Newton_step(lamb,delta[n:n+m],s,delta[n+m:])

        # Update substep
        z = z+0.95*alpha*delta

        # Stop criterion
        if (np.linalg.norm(-b[:n]) < epsilon) or (np.linalg.norm(-b[n:n+m]) < epsilon) or (np.abs(mu) < epsilon):
            break

        # Update M_KKT
        x = z[:n]
        lamb = z[n:n+m]
        s = z[n+m:]
        M,Lamb,S = createMatrix(lamb,s)

    end_time = time.time()

    print('\nProblem dimension:',n)
    print('Found minimum:',f(x))
    print('Actual minimum:',f(-g))
    print('Iterations:',i)
    print("Computation time (sec): ",end_time-start_time)


if __name__ == '__main__':

    # Read problem dimension (3 by default)
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
