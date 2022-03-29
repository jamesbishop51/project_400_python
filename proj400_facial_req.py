#! /usr/bin/python

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import json
import requests
import datetime
from picamera import PiCamera
from time import sleep
import os
import boto3
import google.cloud
#import pyrebase

#Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
#Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"
serverToken = 'AAAA7dEceEA:APA91bHVgUC2f3rRAqVY0GpNA-FJFo5MhMQa20SW4bk_ZWonKtPC_JGAB2qvI8EC6MSk7SBWRo4GZ9yoZ9PR46c86DJp8DNf2yC21GvXxfd6B_equtfej4wMANs5hrPI3jIqX3vQEUub'
phonetoken = 'eUMUuLwQRGK5m4JGFDNXV1:APA91bFltnhW4rbqMBPdOJc3o52VY2QJd669c1A1H0K-U8jbNU1TaorB3Y9JAebznhmo5WyP1sw6qme9CK_YcJvHZ17f4cA3whkQbG-4Uutp1bOOnc825ZQRjgUlVxAR2PFtVrE8S-_E'
ct = datetime.datetime.now()
ct = str(ct)

s3 = boto3.client('s3')

#client = storage.Client()

#bucket = client.get_bucket('project-400-196b2.appspot.com')
#firebaseConfig = {
   # 'apiKey': "AIzaSyAx9vbHie88_571eN-d3svj18MmpxGvhj4",
    #'authDomain': "project-400-196b2.firebaseapp.com",
    #'databaseURL': "https://project-400-196b2-default-rtdb.firebaseio.com",
    #'projectId': "project-400-196b2",
    #'storageBucket': "project-400-196b2.appspot.com",
    #'messagingSenderId': "1021415553088",
    #'appId': "1:1021415553088:web:047b4db88a8c1d4b15c7a5",
    #'measurementId': "G-SPGRVP8EG2"

#}
#firebase = pyrebase.initialize_app(firebaseConfig)

#storage = firebase.storage()

#camera = PiCamera()
# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# initialize the video stream and allow the camera sensor to warm up
# Set the ser to the followng
# src = 0 : for the build in single web cam, could be your laptop webcam
# src = 2 : I had to set it to 2 inorder to use the USB webcam attached to my laptop
#vs = VideoStream(src=2,framerate=10).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# start the FPS counter
fps = FPS().start()

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	# Detect the fce boxes
	boxes = face_recognition.face_locations(frame)
	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(frame, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown" #if face is not recognized, then print Unknown

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)

			#If someone in your dataset is identified, print their name on the screen
			if currentname != name:
				currentname = name
				print(currentname)
				
				headers = {
                    'Content-Type': 'application/json',
        'Authorization': 'key=' + serverToken,
        }
				
				body = {
          'notification': {'title': 'Person:' + name,
                            'body': 'was detected at :' + ct
                            },
          'to':
              phonetoken,
          'priority': 'high',
        #   'data': dataPayLoad,
        }
				
				response = requests.post("https://fcm.googleapis.com/fcm/send",headers = headers, data=json.dumps(body))
				img_name = name + ' ' + ct + '.jpg'
				cv2.imwrite(img_name, frame)
				#zebraBlob = bucket.get_blob(img_name)
				#zebraBlob.upload_from_filename(filename='/home/pi/facial_recognition/'+img_name)
				s3.upload_file(img_name,'project4007bd471b522aa4fd58403d33e6556c679133629-dev',"public/"+img_name)
				print('Taking a picture.')
                
                
                        
		# update the list of names
		names.append(name)

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image - color is in BGR
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 225), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (0, 255, 255), 2)

	# display the image to our screen
	cv2.imshow("Facial Recognition is Running", frame)
	key = cv2.waitKey(1) & 0xFF

	# quit when 'q' key is pressed
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
