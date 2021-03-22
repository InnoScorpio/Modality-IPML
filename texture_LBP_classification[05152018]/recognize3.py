# USAGE
# python classify_med.py --images med_dataset/images --docs med_dataset/docs

# import the necessary packages
#from __future__ import print_function
from feature.localbinarypatterns import LocalBinaryPatterns
#from feature.rgbhistogram import RGBHistogram
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import argparse
import glob
import cv2
import paths1
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required = True,
	help = "path to the image dataset")
ap.add_argument("-m", "--docs", required = True,
	help = "path to the document dataset")
args = vars(ap.parse_args())

#for imagePath in paths1.list_images(args["images"]):
	#print(imagePath)

#for docPath in paths1.list_images(args["docs"]):
	#print(docPath)
	
# grab the image and doc paths
#imagePaths = sorted(glob.glob(args["images"] + "/*.jpg"))
#docPaths = sorted(glob.glob(args["docs"] + "/*.txt"))

imagePaths = sorted(paths1.list_images(args["images"]))
docPaths = sorted(paths1.list_images(args["docs"]))
	
# initialize the list of data and class label targets
data = []
target = []

# initialize the image descriptors
#desc = RGBHistogram([8, 8, 8])
desc = LocalBinaryPatterns(24, 8)

# loop over the image and mask paths
for (imagePath, docPath) in zip(imagePaths, docPaths):
	# load the image and mask
	
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#LBP feature
	features = desc.describe(gray)
		
	#mask = cv2.imread(maskPath)
	#mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	# describe the image
	#features = desc.describe(image, mask)
	
	# label and data lists
	path_list = imagePath.split(os.sep)
	#print path_list[-2]
	target.append(path_list[-2])
	data.append(features)
		
	#target.append(imagePath.split("_")[-2])

# grab the unique target names and encode the labels
targetNames = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)

# construct the training and testing splits
(trainData, testData, trainTarget, testTarget) = train_test_split(data, target,
	test_size = 0.3, random_state = 42)

# train the classifier
model = RandomForestClassifier(n_estimators = 25, random_state = 84)
model.fit(trainData, trainTarget)

# evaluate the classifier
print(classification_report(testTarget, model.predict(testData),
	target_names = targetNames))

# loop over a sample of the images
for i in np.random.choice(np.arange(0, len(imagePaths)), 10):
	# grab the image and mask paths
	imagePath = imagePaths[i]
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#LBP feature
	features = desc.describe(gray)
	
	# predict what type of flower the image is
	flower = le.inverse_transform(model.predict([features]))[0]
	print(imagePath)
	print("I think this image is a {}".format(flower.upper()))
	cv2.imshow("image", image)
	cv2.waitKey(0)

