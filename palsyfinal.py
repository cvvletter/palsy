"""
Created on Wed Jul  1 19:14:39 2020

@author: nsourlos
"""

#Facial Palsy

#Automatically detect face and extract landmarks

# USAGE
# python palsyfinal.py --shape-predictor patients_landmarks.dat --image images/image005.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import dlib
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
 	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
 	help="path to input image")
args = vars(ap.parse_args())


imname=args["image"] #Set a new variable with the name of the image
imtype=imname #Set a second variable with the name of the image
imtype=imtype[-4:] #Keep the type of image (eg. png or jpg)   
imname=imname[:-4] #Keep the name of the image without its type

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Get shape of the image and create a new empty image with the same dimensions but only with white pixels
#The blank image will give the landmarks in a white background
height, width, channels = image.shape
blank_image = 255*np.ones((height,width,channels), np.uint8)

# detect faces in the grayscale image
rects = detector(gray, 1) #coordinates of rectangles in the image - 
#each rectangle is specified by its top left and its bottom right coordinates

for i,rect in enumerate(rects): #Loop over detected faces
    # determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy array
    
    shape=predictor(gray,rect) #Predict the landmarks based on the grayscale version of the image
    shape=face_utils.shape_to_np(shape) #Convert landmark coordinates to numpy array instead of dlib format
    np.save(imname,shape) #Save the numpy array as an 'npy' file
    
    (x,y,w,h)= face_utils.rect_to_bb(rect) #Get the top left coordinates of face box along with its width and height
    
    #Draw the rectangles and overlap them in the original and in the blank image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(blank_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    #Put a title in the box
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(blank_image, "Face #{}".format(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #Loop over the predicted landmarks and put a tiny circle at each of their locations for original and blank image
    for (x,y) in shape:
        cv2.circle(image, (x,y),1,(0,0,255),-1)
    
    for (x,y) in shape:
        cv2.circle(blank_image, (x,y),1,(0,0,255),-1)


box1=list(face_utils.rect_to_bb(rect)) #Convert the top left coordinates of face box along with its width and height to a list
#Below the four coordinate represent the top left and bottom right corner of the box and not its width and height
box1[2]=box1[2]+box1[0] 
box1[3]=box1[3]+box1[1]
 
np.save(imname+"_box" ,box1) #Save box coordinates as numpy array
    
# show the output image with the face detections + facial landmarks and save them
#cv2.imshow("Output", blank_image)
#cv2.imshow("Output", image)
cv2.imwrite(imname+"_blank"+imtype, blank_image) #".png"
cv2.imwrite(imname+"_"+str(i)+"_land"+imtype, image)  #".png"
# cv2.waitKey(0)
    
