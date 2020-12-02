import cv2 
import os

def check(id):
	isInt = False
	try:
		n = int(id[7:len(id) - 1])
		if n > 0: isInt = True
		else: isInt = False
	except Exception as e:
		isInt = False
	finally:
		return id[0:2] == "BT" and int(id[2:4]) >= 16 and (id[4:7] == "CSE" or id[4:7] == "ECE") and isInt

face_id = input("\nEnter ID (in uppercase): ")

if not check(face_id):
	print("\nError: Entered ID is invalid, program terminated\n")

else:
	cam = cv2.VideoCapture(0) # 0 value = using default webcam
	face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	count = 0
	while True:
		ret, img = cam.read()
		if ret:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting image to gray scale
			faces =  face_detector.detectMultiScale(gray, 1.5, 5)
			for (x, y, w, h) in faces:
				cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
				count += 1
				cv2.imwrite("dataset/"+face_id+"_"+str(count)+'.jpg', gray[y:y+h, x:x+w])
				cv2.imshow('camera_view', img)
				cv2.waitKey(50) #shows a image for 50ms
		if count >= 30:	break
	cam.release()
	cv2.destroyAllWindows()