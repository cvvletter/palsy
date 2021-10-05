"""
Created on Sep 20 2021

@author: cvvletter
"""

# Combine all annotated images into one dataset used for machine learning diagnosis

# import the necessary packages
import numpy as np
import glob
import os
import cv2
import math

def best_fit(X, Y): # Function that finds the best line that finds some points

    xbar = sum(X)/len(X) #Find average value of x coordinates
    ybar = sum(Y)/len(Y) #Find average value of y coordinates
    n = len(X) # or len(Y) #Save in a variable the number of points

    # The two variables below will be used for the estimation of the slope of the line and of its constant
    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = float(numer) / denum #Calculation of the constant of the line
    a = ybar - b * xbar #Calculation of the slope
    #print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b)) #Print returned parameters

    return a, b

def rotate(origin, point, angle): 
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy) 
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

# path to images
path = "/home/cvvletter/Documents/palsy/palda_dataset"
# toggle this to show (1) or don't show (0) images
show_images = 0

# this list is used to hold all the landmarks
features = []
# and this list is used to hold all the diagnoses
labels = []
# this makes it so that labels[i] contains the diagnosis corresponding to the landmarks contained in data[i]

# add the landmark data for all peripheral palsy patients to the dataset
# counter = 0
for file in glob.glob(path + "/peripheral/*.npy"):
    if not file.endswith("_box.npy") and file.endswith(".npy"):
        landmarks = np.load(file)
        # counter += 1
        # print("Landmark size of Peripheral picture {} : {}", counter, np.shape(landmarks))
        features.append(landmarks)
        labels.append("periphl") # add the correct label to a separate array

# add the landmark data for all central palsy patients to the dataset
for file in glob.glob(path + "/central/*.npy"):
    if not file.endswith("_box.npy") and file.endswith(".npy"):
        landmarks = np.load(file)
        features.append(landmarks)
        labels.append("central") # add the correct label to a separate array

# add the landmark data for all healthy people to the dataset
newpath = path + "/Healthy/"
for filename in os.listdir(newpath):
    # read file
    file = os.path.join(newpath,filename)
    if os.path.isfile(file) and '_box.npy' not in file:
        continue
    elif os.path.isfile(file) and '_box.npy' in file:
        facebox = np.load(file)
        landmarks = np.load(file[:-8]+'.npy')
        image = cv2.imread(file[:-8]+'.jpg')
        lasttype = '.jpg'
        if image is None:
            image = cv2.imread(file[:-8]+'.png')
            lasttype = '.png'

        # show image (for validation only)
        if (show_images):
            temp = image.copy()
            cv2.imshow("Image", temp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # show image with landmarks (for validation only)
        if (show_images):
            temp = image.copy()
            for (x,y) in landmarks:
                cv2.circle(temp, (x,y),1,(0,0,255),-1)
            cv2.imshow("Image With Landmarks", temp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # crop image and landmarks
        image_cropped = image[facebox[1]:facebox[3],facebox[0]:facebox[2]] #Crop image in x and y direction to include the face only  
        landmarks[:,0] -= facebox[0]
        landmarks[:,1] -= facebox[1]

        # show cropped image (for validation only)
        if (show_images):
            temp = image_cropped.copy()
            cv2.imshow("Cropped Image", temp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # show cropped image with cropped landmarks (for validation only)
        if (show_images):
            temp = image_cropped.copy()
            for (x,y) in landmarks:
                cv2.circle(temp, (x,y),1,(0,0,255),-1)
            cv2.imshow("Cropped Image With Landmarks", temp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # resize image
        image_resized = cv2.resize(image_cropped,(900,900))

        # resize landmarks
        size_initial = image_cropped.shape
        size_rescale = image_resized.shape
        landmarks[:,0] = landmarks[:,0] * (size_rescale[1]/size_initial[1])
        landmarks[:,1] = landmarks[:,1] * (size_rescale[0]/size_initial[0])
        landmarks = landmarks.astype(int)

        # show resized image (for validation only)
        if (show_images):
            temp = image_resized.copy()
            cv2.imshow("Resized Image", temp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # show cropped image with cropped landmarks (for validation only)
        if (show_images):
            temp = image_resized.copy()
            for (x,y) in landmarks:
                cv2.circle(temp, (x,y),1,(0,0,255),-1)
            cv2.imshow("Resized Image With Landmarks", temp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # calculations for rotation so the eyes are aligned
        # todo: rewrite so it is more clear
        a = landmarks.copy() # shorthand for landmarks
        centre_left_eye=np.abs(((a[37]+a[38]+a[40]+a[41]+a[36]+a[39])/6)) #Calculate centre of the left eye (average)
        centre_right_eye=np.abs(((a[43]+a[44]+a[46]+a[47]+a[45]+a[42])/6)) #Calculate centre of the right eye (average)
        dY = centre_left_eye[1] - centre_right_eye[1] #Calculate the y centre of the average of the two eyes
        dX = centre_left_eye[0] - centre_right_eye[0] #Calculate the x centre of the average of the two eyes
        angle = np.degrees(np.arctan2(dY, dX)) - 180 # calculate the angle in which we will rotate face so that eyes are aligned
        eyesCenter = tuple(i for i in (centre_right_eye+centre_left_eye)/2) # Compute center (x, y)-coordinates (i.e., the median point) between the two eyes in the input image
        # newpoint=np.zeros((len(a),2)) # Create an array of size 68*2 to add new landmarks in it after rotation
        for i in range(len(a)):
            c=tuple(i for i in a[i]) #Make its landmark position a tuple which is needed as input to the rotation function
            a[i]=rotate(eyesCenter,c,math.radians(-angle)) #Rotate landmarks using function defined above
        a=a.astype(int) # Make new landmark positions integer values since rotation results in float values  
        M = cv2.getRotationMatrix2D(eyesCenter, angle, 1) # Grab the rotation matrix for rotating and scaling the face which will be used in image rotation
        sh=image_resized.shape[0:2] #Get only width and height and not the channels
        image_rotated = cv2.warpAffine(image_resized, M, sh, flags=cv2.INTER_CUBIC) #Apply the matrix transformation to get final image
        landmarks = a.copy()

        # show rotated image(for validation only)
        if (show_images):
            temp = image_rotated.copy()
            cv2.imshow("Rotated Image", temp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # show rotated image with cropped landmarks (for validation only)
        if (show_images):
            temp = image_rotated.copy()
            for (x,y) in landmarks:
                cv2.circle(temp, (x,y),1,(0,0,255),-1)
            cv2.imshow("Rotated Image With Landmarks", temp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # add the transformed landmarks to the list of healthy landmarks
        features.append(landmarks)
        labels.append("healthy")

features = np.reshape(features,(len(features),68*2))
np.savetxt('features.txt', features, "%3d")
np.savetxt('labels.txt', labels, "%s" )
