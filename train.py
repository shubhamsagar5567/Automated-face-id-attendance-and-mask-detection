import cv2
import numpy as np
import os

path = 'dataset' # Path for face image database
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def give_ID(imagePath):
	global path
	imgFileName = imagePath[len(path) + 1::]
	return imgFileName.split('_')[0]

def convert_ID_to_num(id):
	numId = ""
	for x in list(id):
		if x == 'B' or x == 'T' or x == 'S' or x == 'E': numId += '1'
		elif x == 'C': numId += '2'
		else: numId += x
	return int(numId)

# function to get the images and label data
def getImagesAndLabels(path):
	imagePaths = [os.path.join(path, f) for f in os.listdir(path)] 
	faces, ids = [], []
	for imagePath in imagePaths:
		img_numpy = cv2.imread(imagePath, cv2.COLOR_BGR2GRAY)
		faces.append(img_numpy)
		id = give_ID(imagePath)
		ids.append(convert_ID_to_num(id))
	return faces, ids

faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
# Print the numer of faces trained and end program
print("\n>> ", len(np.unique(ids)), " faces trained")