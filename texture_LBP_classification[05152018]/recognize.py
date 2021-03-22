# USAGE
# python recognize.py --training MedDB5000/training --testing MedDB5000/testing

# import the necessary packages
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from pyimagesearch import dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.colordescriptor import ColorDescriptor
from sklearn.ensemble import RandomForestClassifier
from pyimagesearch.hog import HOG
from sklearn.metrics import classification_report
from pyimagesearch.rgbhistogram import RGBHistogram
from pyimagesearch.searcher import Searcher
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import paths
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
	help="path to the training images")
ap.add_argument("-e", "--testing", required=True, 
	help="path to the tesitng images")
args = vars(ap.parse_args())

# initialize the local binary patterns descriptor along with
# the data and label lists
hog = HOG(orientations = 18, pixelsPerCell = (10, 10),
	cellsPerBlock = (1, 1), transform = True)
	
lbp = LocalBinaryPatterns(24, 8)

rgb = RGBHistogram([8, 8, 8])

cd  = ColorDescriptor([8, 12, 3])

data = []
labels = []



# loop over the training images
for imagePath in paths.list_images(args["training"]):
	# load the image, convert it to grayscale, and describe it
	#print(imagePath)
	# load the image, convert it to grayscale, and describe it
	#print(imagePath)
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (100, 100)) 
	image = cv2.imread(imagePath)
	# Resize image to 100 x 100 
	image = cv2.resize(image, (100, 100)) 
		
	# RGB Histogram
	rgb_hist = rgb.describe(image)
		
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	#LBP feature
	lbp_hist = lbp.describe(gray)
		
	gray = dataset.deskew(gray, 100)
	gray = dataset.center_extent(gray, (100, 100))
		
	#HOG feature	
	hog_hist = hog.describe(gray)
	
	# describe the color of 5 regions in image
	features = cd.describe(image)
	
	x = rgb_hist.tolist()
	y = lbp_hist.tolist()
	z = hog_hist.tolist()
	x.extend(y)
	x.extend(z)
	combined =np.array(x)
	data.append(combined)
	#data = data.shape(1, -1)
# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42)
model.fit(np.array(data), np.array(labels))

# loop over the testing images
for imagePath in paths.list_images(args["testing"]):
	# load the image, convert it to grayscale, describe it,
	# and classify it
    '''
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (100, 100)) 
    image = cv2.imread(imagePath)
	# Resize image to 100 x 100 
    image = cv2.resize(image, (100, 100)) 
		
	# RGB Histogram
    rgb_hist = rgb.describe(image)
		
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	#LBP feature
    lbp_hist = lbp.describe(gray)
		
    gray = dataset.deskew(gray, 100)
    gray = dataset.center_extent(gray, (100, 100))
		
	#HOG feature	
    hog_hist = hog.describe(gray)
	
	# describe the color of 5 regions in image
    features = cd.describe(image)
    ''' 
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cd.describe(features)
    hist = hist.reshape(1,-1)
    labels.append(hist)
    prediction = model.predict(hist)[0]
    print(prediction)
  
	
	#print(model.score(trainData))
  
	# display the image and the prediction
    cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)