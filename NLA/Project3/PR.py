import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import time
from scipy.sparse.linalg import spsolve

#Function that creates our desired diagonal matrix with 1/n if nj != 0 and 0 otherwise
def diagonal(matrix):
	with np.errstate(divide='ignore', invalid='ignore'):
		d = np.divide(1.,np.asarray(matrix.sum(axis=0)).reshape(-1))
	#Remove NaN's and Inf's
	np.ma.masked_array(d, ~np.isfinite(d)).filled(0)
	#Construct a sparse matrix from diagonals.
	return sp.diags(d)

# Power Method for PageRank computation storing matrices
def PM_PR(A,m,tol):
	n = A.shape[0]
	tic = time.time()
	e = np.ones((n,1))
	zj = np.asarray([m/n if A[:,i].count_nonzero()>0 else 1./n for i in range(n)])
	
	xk = np.ones((n,1))
	xk1=  np.ones((n,1))/n
	
	while np.linalg.norm(xk1-xk,np.inf) > tol:
		xk = xk1
		zxk = zj.dot(xk)
		xk1 = (1-m)*A.dot(xk)+e*zxk
	print("Execution time of the power method for PR:",time.time()-tic,"seconds")
	#Normalitzation
	xk1 = xk1/np.sum(xk1)
	return xk1

# Function that creates a dictionary of indices used in PM_PR_NoStoring. Also called "L"
def makeIndices(rows,cols):
	indices = {}
	for i in range(len(cols)):
		colIndex = cols[i]
		if colIndex in indices:
			indices[colIndex] = np.append(indices[colIndex],rows[i])
		else:
			indices[colIndex] = np.asarray([rows[i]])
	return indices

#Power method for PageRank computation without storing matrices
def PM_PR_NoStoring(matrix,m,tol):
	n = matrix.shape[0]
	tic = time.time()
	rows = matrix.nonzero()[0]
	cols = matrix.nonzero()[1]
	indices = makeIndices(rows,cols)
	
	x = np.ones((n,1))/n
	xc = np.ones((n,1))
	while (np.linalg.norm(x-xc,np.inf)>tol):
		#Teacher's code
		xc = x
		x = np.zeros((n,1))
		for j in range(n):
			if j in indices:
				if len(indices[j]) != 0:
					x[indices[j]] += xc[j]/len(indices[j])
				else:
					x += xc[j]/n
			else:
				x += xc[j]/n
		x = (1-m)*x + m/n
	print("Execution time of the power method for PR without storing matrices:",time.time()-tic,"seconds")
	#Normalitzation
	return x/np.sum(x)


def systemPR(A,m):
	n = A.shape[0]
	tic = time.time()
	I = sp.identity(n)
	x = spsolve(I-(1-m)*A, np.ones((n,1)))
	print("Execution time of the Linear solver of PageRank vector",time.time()-tic,"seconds")
	print(x/np.sum(x))


if __name__ == '__main__':

	name = -1
	while name != 2 and name != 1:
		name = int(input("What matrix do you want to use?\n 1. 'p2p-Gnutella30.mtx'\n 2. 'p2p-Gnutella31.mtx'\n"))
		if name == 1: matrix = sio.mmread("p2p-Gnutella30.mtx")
		elif name == 2: matrix = sio.mmread("p2p-Gnutella31.mtx")
		else:
			print("Please input 1 or 2.")
			
	#Sparse
	matrix = sp.csr_matrix(matrix)
	#D creation
	D = diagonal(matrix)
	A = sp.csr_matrix(matrix.dot(D))

	tol = -1
	while tol == -1:
		tol = float(input("Input the desired tolerance, for example, '1e-5'\n"))

	dampling = -1
	while dampling == -1:
		dampling = float(input("Input the desired dampling factor, for example, '0.15'\n"))

	
	end = "y"
	while end == "y":
		method = -1
		while method != 0 and method != 1 and method != 2:
			method = int(input("What method do you want to use?\n 0. Power Method with storing\n 1. Power Method without storing\n 2. Linear solver\n"))

		if method == 0:
			print("Starting the computation with storing...")
			xk1 = PM_PR(A,dampling,tol)
			print("Normalized Page Rank Vector found:\n",xk1)

		elif method == 1:
			x = PM_PR_NoStoring(matrix,0.15,1e-5)
			print("Normalized Page Rank Vector found:\n",x)
		elif method == 2:
			print("Starting the computation as a Linear system... (this will take some time)")
			systemPR(A,dampling)
		end = input("Do you want to try other method with the same parameters? ('y' or 'n')\n")
