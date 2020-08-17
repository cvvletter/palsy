"""
Created on Tue Mar 24 15:53:33 2020

@author: nsourlos
"""

#Check if manual annotation of landmarks was successful

# USAGE
# python pointscheck.py --path-location /home/user/Desktop/images 
# The above folder should only contain images and their 'npy' landmark files with the same name as the images
# For example, 'im01.npy' and 'im01.jpg'
# The only formats of the images that are allowed are jpg and png. The 'npy' files should be 68*2

# import the necessary packages
import cv2
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--path-location", required=True,
 	help="path to folder in which manual annotated images will be checked")
args = vars(ap.parse_args())

#Function that given an image and its 68 landmarks it creates lines that correspond to face areas (eg. eyes, mouth etc)
def lines(image, array):
    #Need to make the numpy array tuple in order to use it below
    array_of_tuples = map(tuple, array) 
    tuple_of_tuples = tuple(array_of_tuples)
    
    for i in range(len(array)): #Loop over landmarks and create lines
        if i<16: #Chin
            cv2.line(image, tuple_of_tuples[i], tuple_of_tuples[i+1], [0, 255, 0], 2) 
        elif i>16 and i<21: #Right Brow
            cv2.line(image, tuple_of_tuples[i], tuple_of_tuples[i+1], [0, 255, 0], 2) 
        elif i>21 and i<26: #Left Brow
            cv2.line(image, tuple_of_tuples[i], tuple_of_tuples[i+1], [0, 255, 0], 2)
        elif i>26 and i<35: #Nose
            cv2.line(image, tuple_of_tuples[i], tuple_of_tuples[i+1], [0, 255, 0], 2)
        elif i==35: 
            cv2.line(image, tuple_of_tuples[i], tuple_of_tuples[i-5], [0, 255, 0], 2)
        elif i>35 and i<41: #Right Eye
            cv2.line(image, tuple_of_tuples[i], tuple_of_tuples[i+1], [0, 255, 0], 2)
        elif i==41:
                cv2.line(image, tuple_of_tuples[i], tuple_of_tuples[i-5], [0, 255, 0], 2)
        elif i>41 and i<47: #Left Eye
            cv2.line(image, tuple_of_tuples[i], tuple_of_tuples[i+1], [0, 255, 0], 2)
        elif i==47:
            cv2.line(image, tuple_of_tuples[i], tuple_of_tuples[i-5], [0, 255, 0], 2)
        elif i>47 and i<59: #Outer part of mouth
            cv2.line(image, tuple_of_tuples[i], tuple_of_tuples[i+1], [0, 255, 0], 2)
        elif i==59:
            cv2.line(image, tuple_of_tuples[i], tuple_of_tuples[i-11], [0, 255, 0], 2)
        elif i>59 and i<67: #Inner part of mouth
            cv2.line(image, tuple_of_tuples[i], tuple_of_tuples[i+1], [0, 255, 0], 2)
        elif i==67:
            cv2.line(image, tuple_of_tuples[i], tuple_of_tuples[i-7], [0, 255, 0], 2)


path = args["path_location"] #Get the path location
os.chdir(path) #Go to that path

for filename in os.listdir(path+'/'): #Loop over files in the path
    #Get the filename and then set two new variables with the name only and with the type only
    filname=filename 
    imname=filname
    imtype=filname
    imname=imname[:-4]
    imtype=imtype[-4:]
    if os.path.isfile(os.path.join(path,filename)) and 'npy' in filename: #If filename is a 'npy' file then:
        lands=np.load(filname) #Load landmark positions
        im=cv2.imread(filname[:-4]+".jpg") #Load image assuming that its type is jpg
        if im is None: #If not jpg
            im=cv2.imread(filname[:-4]+".png") #Try to load it as png
            
        print(filename) #Print npy file that is being processed
        im=cv2.resize(im,(900,900)) #Resize it to 900*900 as when we manually annotated it
        height, width, channels = im.shape #Get its shape (dimensions)
        blank_image = 255*np.ones((height,width,channels), np.uint8) #Create a blank image with only white pixels
        #It has the same dimensions as the original as it is used to project the landmarks in a white background
        init=im.shape #Set another variable with the shape of the image used below
        im=cv2.resize(im,(280,420)) #Resize image to smaller size
        blank_image=cv2.resize(blank_image,(280,420)) #Same for 'blank' image
        fin=im.shape #Get the shape of the new resized image
        
        #Linear transform of the landmarks in order to overlap at the same points in the resized image
        lands[:,0]=lands[:,0]*fin[1]/init[1]
        lands[:,1]=lands[:,1]*fin[0]/init[0]
        lands=lands.astype(int) #Since the transformation results in float values and we need integers

        for (x,y) in lands: #Loop over the transformed landmark positions
            #Add circles in the original and 'blank' images
            cv2.circle(im, (x,y),1,(0,0,255),-1)
            cv2.circle(blank_image, (x,y),1,(0,0,255),-1)
 
        #Save the images with their circles in png format
        cv2.imwrite(imname+"_blank"+".png", blank_image) #".png"
        cv2.imwrite(imname+"_land"+".png", im)  #".png"
        
        lines(im,lands) #Use above function to draw lines
        cv2.imwrite(imname+"_land_lines"+".png", im)  #Save the new image with the lines in png format
        
        lines(blank_image,lands) #Create another image with lines in white background 
        cv2.imwrite(imname+"_blank_lines"+".png", blank_image) #Save it as well
