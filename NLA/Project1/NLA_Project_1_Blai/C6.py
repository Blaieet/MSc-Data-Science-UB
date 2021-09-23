import numpy as np
import sys
import time
from scipy.linalg import ldl

# Reads an array from file 'filename'
def readMatrix(filename,shape,symmetrical=False):
    m = np.loadtxt(filename)
    M = np.zeros(shape)

    if type(shape) == int:
        for i in range(m.shape[0]):
            M[int(m[i,0])-1] = m[i,1]
    else:
        for i in range(m.shape[0]):
            M[int(m[i,0])-1,int(m[i,1])-1] = m[i,2]
            if symmetrical:
                M[int(m[i,1])-1,int(m[i,0])-1] = m[i,2]
    return M


# Creates the M_KKT, Lambda and S matrices
def createMatrix(lamb,s):
    Lamb = np.diag(lamb)
    S = np.diag(s)
    row1 = np.concatenate((G,-A,-C),axis=1)
    row2 = np.concatenate((-np.transpose(A),np.zeros((p,p+m))),axis=1)
    row3 = np.concatenate((-np.transpose(C),np.zeros((m,p)),np.diag(-1/lamb*s)),axis=1)
    M = np.concatenate((row1,row2,row3))
    return M,S,Lamb


# Original function, f(x)
def f(x):
    return 0.5*np.transpose(x).dot(G).dot(x)+np.transpose(g).dot(x)


# Function F(z)
def fun(x,gamma,lamb,s):
    comp1 = G.dot(x)+g-A.dot(gamma)-C.dot(lamb)
    comp2 = b-np.transpose(A).dot(x)
    comp3 = s+d-np.transpose(C).dot(x)
    comp4 = s*lamb
    return np.concatenate((comp1,comp2,comp3,comp4))


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

    global lamb,s,x,z,gamma
    start_time = time.time()

    # Create M_KKT, S and Lambda matrices
    M,S,Lamb = createMatrix(lamb,s)

    for i in range(maxIter):
        lamb_inv = np.diag(1/lamb)

        # Standard Newton step (solve KKT system)...
        b = fun(x,gamma,lamb,s)
        r1, r2, r3, r4 = b[:n], b[n:n+p], b[n+p:n+p+m], b[n+p+m:]
        b = np.concatenate(([-r1,-r2,-r3+1/lamb*r4]))

        # ...using LDL^T factorization
        L, D, perm = ldl(M)
        y = np.linalg.solve(L,b)
        delta = np.linalg.solve(D.dot(np.transpose(L)),y)
        delta_s = lamb_inv.dot(-r4-s*delta[n+p:])
        delta = np.concatenate((delta,delta_s))

        # Step-size correction substep
        alpha = Newton_step(lamb,delta[n+p:n+p+m],s,delta[n+m+p:])

        # Compute correction parameters
        mu = s.dot(lamb)/m
        muTilde = ((s+alpha*delta[n+m+p:]).dot(lamb+alpha*delta[n+p:n+m+p]))/m
        sigma = (muTilde/mu)**3

        # Corrector substep...
        Ds_Dlamb = np.diag(delta[n+p+m:]*delta[n+p:n+p+m])
        b = np.concatenate((-r1, -r2, -r3+lamb_inv.dot(r4+Ds_Dlamb.dot(e)-sigma*mu*e)))

        # ...using LDL^T factorization again
        y = np.linalg.solve(L,b)
        delta = np.linalg.solve(D.dot(np.transpose(L)),y)
        delta_s = lamb_inv.dot(-r4-Ds_Dlamb.dot(e)+sigma*mu*e-s*delta[n+p:])
        delta = np.concatenate((delta,delta_s))

        # Step-size correction substep
        alpha = Newton_step(lamb,delta[n+p:n+p+m],s,delta[n+m+p:])

        # Update substep
        z = z+0.95*alpha*delta

        # Stop criterion
        if (np.linalg.norm(-r1) < epsilon) or (np.linalg.norm(-r2) < epsilon) or (np.linalg.norm(-r3) < epsilon) or (np.abs(mu) < epsilon):
            break

        # Update M_KKT
        x = z[:n]
        gamma = z[n:n+p]
        lamb = z[n+p:n+m+p]
        s = z[n+m+p:]
        M,Lamb,S = createMatrix(lamb,s)

    end_time = time.time()

    print('\nProblem dimension:',n)
    print('Found minimum: f(x)=',f(x))
    print('Iterations:',i)
    print("Computation time (sec): ",end_time-start_time)


if __name__ == '__main__':

    # Read folder with problem data
    if len(sys.argv) == 1:
        print("You can change the problem indicating the folder name ('optpr1' or 'optpr2'), defualt 'optpr1'.")
        folder = "optpr1"
    else:
        folder = sys.argv[1]

    # Problem dimensions
    n = int(np.loadtxt(folder+"/G.dad")[:,0][-1])
    p = int(np.loadtxt(folder+"/b.dad")[:,0][-1])
    m = int(np.loadtxt(folder+"/d.dad")[:,0][-1])

    # Read data
    A = readMatrix(folder+"/A.dad",(n,p))
    b = readMatrix(folder+"/b.dad",p)
    C = readMatrix(folder+"/C.dad",(n,m))
    d = readMatrix(folder+"/d.dad",m)
    e = np.ones((m))
    G = readMatrix(folder+"/G.dad",(n,n),symmetrical=True)
    g = readMatrix(folder+"/g.dad",n)
    x = np.zeros((n))
    gamma = np.ones((p))
    lamb = np.ones((m))
    s = np.ones((m))
    z = np.concatenate((x,gamma,lamb,s))

    # Call algorithm
    Newton(epsilon=10e-15)
