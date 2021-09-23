import numpy as np
import pandas as pd
import sys
from scipy.linalg import solve_triangular, qr
import matplotlib.pyplot as plt

# Program to solve the LS problem using SVD
def ls_svd(A,b):
    return np.linalg.pinv(A).dot(b)

# Program to solve the LS problem using QR
def ls_qr(A,b):
    # rank of matrix A
    r = np.linalg.matrix_rank(A)

    # full-rank matrix
    if r == A.shape[1]:
        Q1,R1 = np.linalg.qr(A)
        y1 = np.transpose(Q1).dot(b)
        x = solve_triangular(R1,y1)

    # rank deficient matrix
    else:
        Q,R,P = qr(A,mode='economic',pivoting=True)  # scipy qr
        R1 = R[:r,:r]
        c = np.transpose(Q).dot(b)[:r]
        u = solve_triangular(R1,c)
        v = np.zeros((A.shape[1]-r))
        x = np.linalg.solve(np.transpose(np.eye(A.shape[1])[:,P]),np.concatenate((u,v)))

    return x

# Compute the LS solution for the dataset in 'datafile'
# k is a list with the degrees of the polynomials to be used for the fit
def ex1(k):
    # read data from dades
    data = []
    f = open("dades", "r")
    lines = f.readlines()
    for line in lines:
        data.append(np.fromstring(line,sep=' '))
    data = np.asarray(data)

    # problem dimension
    n = data.shape[0]

    # create matrix A with for the polynomial with maximum degree
    full_A = np.zeros((n,max(k)+1)) # matrix with max cols
    for i in range(n):
        full_A[i,:] = [data[i,0]**k for k in range(max(k)+1)]

    # create vector b
    b = data[:,1]

    # plot data
    plt.scatter(data[:,0],data[:,1],s=5,c='indigo')
    x = np.linspace(0.8,8.2,100)

    # fit the data with polnomials of different degrees
    for m in k:
        print("\nDegree =", m)
        A = full_A[:,:m+1]
        # compute QR and SVD solutions
        x_qr = ls_qr(A,b)
        x_svd = ls_svd(A,b)
        print("\nLS solution using SVD ->",x_svd)
        print("x_svd norm = %.4f" % np.linalg.norm(x_svd))
        print("Error_svd norm = %.4f" % np.linalg.norm(A.dot(x_svd)-b))
        print("\nLS solution using QR ->",x_qr)
        print("x_qr norm = %.4f" % np.linalg.norm(x_qr))
        print("Error_qr norm = %.4f" % np.linalg.norm(A.dot(x_qr)-b))

        # plot fit
        y = sum(x_svd[i]*x**i for i in range(m+1))
        plt.plot(x,y,label=str(m)+' degree fit')
    plt.legend()
    plt.show()

# Compute the LS solution for the dataset in 'datafile2.csv'
def ex2():
    # read data
    data = pd.read_csv("dades_regressio.csv",header=None)
    A = data.iloc[:,:-1].values
    b = data.iloc[:,-1].values
    print("Shape of A:", A.shape)
    print("Rank of A:", np.linalg.matrix_rank(A))

    # Compute qr and SVD solutions
    x_svd = ls_svd(A,b)
    print("\nLS solution using SVD ->",x_svd)
    print("x_svd norm = %.4f" % np.linalg.norm(x_svd))
    print("Error_svd norm = %.4f" % np.linalg.norm(A.dot(x_svd)-b))
    x_qr = ls_qr(A,b)
    print("\nLS solution using QR ->",x_qr)
    print("x_qr norm = %.4f" % np.linalg.norm(x_qr))
    print("Error_qr norm = %.4f" % np.linalg.norm(A.dot(x_qr)-b))



def main():
    print("=================\nDATASET: datafile\n=================")
    k = []
    if len(sys.argv) == 1:
        print("You can specify different degrees for the polynomials used to fit the data:")
        print("For example, use python3 LS.py 3 4 5 to use polynomials of degree 3, 4 and 5.")
        k.append(3)
    else:
        for degree in sys.argv[1:]:
            k.append(int(degree))

    ex1(k)

    print("\n======================\nDATASET: datafile2.csv\n======================")
    ex2()

if __name__ == '__main__':
    main()
