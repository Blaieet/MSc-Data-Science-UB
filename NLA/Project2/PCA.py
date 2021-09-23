import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import sys

#Function that zuses Correlation or Covariance matrix to cumpute X and Y of the PCA
def matrix(type,M,transpose):
	n = M.shape[0]
	if type == "correlation":
		# using correlation matrix
		X = np.transpose((M - M.mean(axis=0)) / M.std(axis=0))
		Y = 1/np.sqrt(n-1) * np.transpose(X)

	else:
		# using covariance matrix
		if transpose:
			X = M-M.mean(axis=0)
		else:
			X = np.transpose(M-M.mean(axis=0))
		#print(np.mean(X,axis=1))
		Y = 1/np.sqrt(n-1) * np.transpose(X)
		#Cx = np.transpose(Y).dot(Y)
	return X,Y

#Function that performs the PCA with Numpy SVD
def SVD(X,Y):
	U,S,Vt = np.linalg.svd(Y,full_matrices=False)
	return Vt.dot(X),S

#Function that returns the total variance accumulated
def varPortions(S):
	total_var = sum(S**2)
	return [s**2 / total_var for s in S]

#Function that performs the PCA with the Sklearn
def pcaSklearn(M):
	#use sklearn pca
	pca1 = PCA(n_components=4)
	pca1.fit(M)
	print(pca1.explained_variance_ratio_)

#For the exercice 2, function that creates the output file requested
def createFile(df,PCA,var_portions):
	output = pd.DataFrame(data=PCA[:,:20], index=df.columns, columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20'])
	output.index.name = 'Sample'
	output['Variance'] = var_portions
	output.to_csv("PCA_to_RCsGoff.csv")


def ex1(type):
	# Read data from example.dat
	data = []
	f = open("example.dat", "r")
	lines = f.readlines()
	for line in lines:
		data.append(np.fromstring(line,sep=' '))
	M = np.asarray(data)
	n = M.shape[0]
	m = M.shape[1]

	X,Y = matrix(type,M,False)

	PCA,S = SVD(X,Y)

	print("Dataset in the PCA coordinates:")
	print(np.transpose(PCA))

	print("\nStandard deviation of each of the principal components:")
	print(S)

	print("\nPortion of the total variance accumulated in each of the principal components:")
	print(varPortions(S))


	print("\nExtra: PCA total variance accumulated using Sklearn library:")
	pcaSklearn(M)

def ex2(type):
	df = pd.read_csv('RCsGoff.csv')
	df.drop('gene', axis=1, inplace=True)
	M = np.transpose(df.values)
	n = M.shape[0]
	M.shape

	X,Y = matrix(type,M,True)

	PCA,S = SVD(X,Y)

	createFile(df,PCA,varPortions(S))
	print("File 'PCA_to_RCsGoff.csv' created")
	print(pd.read_csv('PCA_to_RCsGoff.csv'))




if __name__ == '__main__':

	type = "covariance"
	if len(sys.argv) == 1:
		print("You can specify the matrix type with 'python PCA.py covariance' or 'python PCA.py correlation'")
		print("Assuming covariance matrix.")
	else:
		type = sys.argv[1]
		if type == "covariance" or type == "correlation":
			print("Using",sys.argv[1],"matrix")
		else:
			print("Matrix type should be inputed as 'correlation' or 'covariance'")
			exit()
	print("====================\nDATASET: Example.dat\n====================")
	ex1(type)

	print("\n======================\nDATASET: RCsGoff.csv\n======================")
	print("Using covariance matrix as requested in the PDF")
	ex2("covariance")
