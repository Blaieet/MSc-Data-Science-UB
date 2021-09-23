import numpy as np
import imageio
import os
import sys

#Returns a grey-scaled image given one that is not
def greyScale(img):
	#Check if its RGB or not
	if img.shape[1] > 1:
		return img[:,:,1]
	return img

#Compresses all images in a given directory using the SVD method
def compress(directory):
	#Check if there's already a path to store the compressed images
	if not os.path.isdir(directory+"/Compressed"):
		#Create it if not
		os.mkdir(directory+"/Compressed")
	#For each file
	for filename in os.listdir(directory):
		#If it's a jpeg
		if filename.endswith(".jpeg"):
			#Get the image name
			imgName = filename.split(".")[0]
			if imgName == "Dog":
				print("Processing Dog.jpeg. This will take 30 secs more than the rest.")
			#Read the image,
			img = greyScale(imageio.imread('images/'+filename))
			#Get its SVD decomposition
			U, sigma, V = np.linalg.svd(img)
			#And start computing an approximation of the image using the first column of U and first row of V
			for i in range(5, 51, 5):
				A = np.matrix(U[:,:i]) * np.diag(sigma[:i]) * np.matrix(V[:i,:])
				#Compute the error
				relative_error = sigma[i]/sigma[0]
				#Construct the name of this new compressed image.
				#Each image will have on its name the percentage of the Frobenius norm captured
				path = directory+"/Compressed/"+imgName+"_"+str(round((1-relative_error)*100,4))+".jpg"
				#Save
				imageio.imwrite(path, np.clip(A,0,255).astype(np.uint8))
	print("All compressed images can be seen in images/Compressed")

if __name__ == '__main__':

	if len(sys.argv) == 1:
		print("You can change the images directory using 'python GraphicsCompression.py [path]'. Assuming: '/images'")
		directory = "images/"
	else:
		directory = sys.argv[1]
	compress(directory)