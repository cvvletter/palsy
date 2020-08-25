"""
Created on Sun Jul  5 11:06:16 2020

@author: nsourlos
"""

#Manual Annotation of Images Tool

# USAGE
# python manual_annot.py --path-location /home/user/Desktop/images 
# The above folder should only contain images and/or directories. Inside directories only images are allowed

# import the necessary packages
import cv2
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--path-location", required=True,
 	help="path to folder in which images will be annotated")
args = vars(ap.parse_args())

#The function below saves the coordinates of the point double clicked on the image
ix,iy = -1,-1 #Set initial values of these variables
def draw_circle(event,x,y,flags,param): # mouse callback function
    global ix,iy #To change the values of these variables based on the following
    if event == cv2.EVENT_LBUTTONDBLCLK: #If left button of mouse is doubleclicked then:
        cv2.circle(img,(x,y),3,(255,0,0),-1) #Create a circle in the image in the clicked point
        ix,iy = x,y #Set the variable values equal to those of the clicked point

path=args["path_location"] #Get the location of the images
os.chdir(path)  #Go in that location
print('Welcome to landmark annotation! Please click a point in the image that appeared to start the annotation')
print('To exit from a specific image hit "Esc". To exit the program at any time press "Ctrl + \\"')
print('To delete the last landmarked clicked press "a"')

for filename in os.listdir(path): #Loop over the filenames contained in that directory
    filname=filename #Save the filename in a variable
    if os.path.isfile(os.path.join(path,filename)) and 'npy' in filename: #If filename is an 'npy' file then continue
        continue
    elif os.path.isdir(os.path.join(path,filename)): #If filename is a directory then:
            for filename2 in os.listdir(path+'/'+filename): #Loop over files of that directory
                 image = cv2.imread(path+'/'+filename+'/'+filename2) #Load image
                 img=cv2.resize(image,(900,900)) #Resize image in order to be 900*900 so that we can annotate points properly
                 cv2.namedWindow('image') #Give a name to the window that contains the image
                 cv2.setMouseCallback('image',draw_circle) #Use the function defined above to create circles in the clicked points
                 arr=[[-1, -1]] #Create a list with lists of the clicked points. The first point is set to [-1, -1] as reference
                 while(1): #While True do the following:
                        cv2.imshow('image',img) #Show the image that we have to annotate
                        k = cv2.waitKey(10) #Wait until a point clicked or a button pressed
                        if k == 27: #If 'Esc' button pressed ('27' key button) then exit
                            print('You stopped the program. No file is created. Continue to next photo/exit')
                            print('If you want to exit entirely click on terminal and then press "Ctrl + \\"')
                            break
                        elif k==ord('a'): #If 'a' pressed then delete the last point clicked (It still remains on the screen)
                            del arr[-1]
                            ix=arr[-1][0]
                            iy=arr[-1][1]
                            print('You deleted the last landmark')
                            print('{:d} points clicked so far out of 68'.format(len(arr)-1))
                        else: #If mouse clicked then: 
                            if arr[-1][0]==ix and arr[-1][1]==iy: #Ignore the first reference point on the list
                                continue
                            elif len(arr)<69: #If the list contains less than 69 elements (68 landmarks + 1 reference) then:
                                #add the coordinates of the clicked point to the list 'arr'
                                if ix!=arr[-1][0] or iy!=arr[-1][1]:
                                    arr.append([ix,iy])
                                    if len(arr)==69: #Set the values to exit if statement
                                        ix=-1
                                        iy=-1
                                    elif len(arr)==68: #Print message that this will be the last point
                                        print('{:d} points clicked so far out of 68'.format(len(arr)-1))
                                        print('Last point to click and then you will be directed to next image/exit')
                                        ix=arr[-1][0]
                                        iy=arr[-1][1]
                                    else: #Print the number of points clicked until now
                                        print('{:d} points clicked so far out of 68'.format(len(arr)-1))
                                        ix=arr[-1][0]
                                        iy=arr[-1][1]
                                else:
                                    continue
                            else:
                                del arr[0] #Delete first point which is the reference
                                print('List is full. Final len is')
                                print(len(arr))
                                fin=np.asarray(arr) #Make the list of lists an array
                                np.save(os.path.join(path,filename+'/'+filename2[:-4]),fin) #Save that array
                                break          
                 cv2.destroyAllWindows() #Close the window and go to the next
                
    else: #Same procedure as above, applied to images in the given folder
        image = cv2.imread(filname) #Load image
        img=cv2.resize(image,(900,900)) #Resize image in order to be 900*900 so that we can annotate points properly
        cv2.namedWindow('image') #Give a name to the window that contains the image
        cv2.setMouseCallback('image',draw_circle) #Use the function defined above to create circles in the clicked points
        arr=[[-1, -1]] #Create a list with lists of the clicked points. The first point is set to [-1, -1] as reference
        while(1): #While True do the following:
            cv2.imshow('image',img) #Show the image that we have to annotate
            k = cv2.waitKey(10) #Wait until a point clicked or a button pressed
            if k == 27: #If 'Esc' button pressed ('27' key button) then exit
                print('You stopped the program. No file is created. Continue to next photo/exit')
                print('If you want to exit entirely click on terminal and then press "Ctrl + \\"')
                break
            elif k==ord('a'): #If 'a' pressed then delete the last point clicked (It still remains on the screen)
                del arr[-1]
                ix=arr[-1][0]
                iy=arr[-1][1]
                print('You deleted the last landmark')
                print('{:d} points clicked so far out of 68'.format(len(arr)-1))
            else: #If mouse clicked then: 
                if arr[-1][0]==ix and arr[-1][1]==iy: #Ignore the first reference point on the list
                    continue
                elif len(arr)<69: #If the list contains less than 69 elements (68 landmarks + 1 reference) then:
                    #add the coordinates of the clicked point to the list 'arr'
                    if ix!=arr[-1][0] or iy!=arr[-1][1]:
                        arr.append([ix,iy])
                        if len(arr)==69: #Set the values to exit if statement
                            ix=-1
                            iy=-1
                        elif len(arr)==68: #Print message that this will be the last point
                            print('{:d} points clicked so far out of 68'.format(len(arr)-1))
                            print('Last point to click and then you will be directed to next image/exit')
                            ix=arr[-1][0]
                            iy=arr[-1][1]
                        else: #Print the number of points clicked until now
                            print('{:d} points clicked so far out of 68'.format(len(arr)-1))
                            ix=arr[-1][0]
                            iy=arr[-1][1]
                    else:
                        continue
                else:
                    del arr[0] #Delete first point which is the reference
                    print('List is full. Final len is')
                    print(len(arr))
                    fin=np.asarray(arr) #Make the list of lists an array
                    np.save(filname[:-4],fin) #Save that array
                    break          
        cv2.destroyAllWindows() #Close the window and go to the next
