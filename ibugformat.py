"""
Created on Tue Jul  7 12:13:44 2020

@author: nsourlos
"""
# Create new text files in the format of the iBUG dataset 

# USAGE
# python ibugformat.py --path-location-images /home/user/Desktop/images  --path-location-text /home/user/Desktop/textfile --path-location-original-text /home/user/Desktop/originaltextfile
# The above folder should only contain images and their 'npy' landmark files with the same name as the images, 
# along with the face boxes in the format 'image01_box.npy'. For example, 'im01.npy', 'im01.jpg' and 'im01_box.npy'
# The only formats of the images that are allowed are jpg and png. The landmark files should be 68*2 for landmarks
# and the box should be an array with 4 elements. 
# The folder should be inside the iBUG folder with the other datasets (eg. helen, afw etc.)
# The originaltext file is the original text file with the iBUG's dataset training files (it is an 'xml' file)
# The final txt file is aved in the image location with name 'traininglast.txt'

# import the necessary packages
import numpy as np
import os
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--path-location-images", required=True,
 	help="path to folder in which images, boxes and landmark file exist")
ap.add_argument("-p", "--path-location-text", required=True,
 	help="path to folder in which the original text file of iBUG format for an image exists")
ap.add_argument("-k", "--path-location-original-text", required=True,
 	help="path to folder in which the original text file of iBUG format for all images exists")
args = vars(ap.parse_args())

path=args["path_location_images"] #Get the location of the images
os.chdir(path) #Go in that directory

for filename in os.listdir(path): #Loop over files of that directory
    if os.path.isfile(os.path.join(path,filename)) and 'npy' in filename: #If filename is a 'npy' file continue
        continue
    elif os.path.isfile(os.path.join(path,filename)) and 'txt' in filename: #If filename is a 'txt' file continue
        continue
    else:
        print(filename) #Print the name of the file that is being processed 
        image = cv2.imread(filename[:-4]+'.jpg') #Load the image
        lastones='.jpg' #Keep the image type
        if image is None: #If image is not jpg then
            image = cv2.imread(filename[:-4]+'.png') #Load image as png
            lastones='.png' #Keep the image type
        resiz=cv2.resize(image,(900,900)) #Resize image to the same size as when manually annotated
        fin=image.shape #Get the dimensions of the final image that we want
        init=resiz.shape #Get the dimensions of the initial image before any transformation
        lands=np.load(filename[:-4]+'.npy') #Load the landmarks file
   
        lands[:,0]=lands[:,0]*fin[1]/init[1] #Transform the x coordinates of the landmarks
        lands[:,1]=lands[:,1]*fin[0]/init[0] #Transform the y coordinates of the landmarks
        lands=lands.astype(int) #Make the landmarks integers since transformation results in float numbers
        np.save(filename[:-4]+'.npy',lands) #Save the new landmarks
        #Just for confirmation
        # for (x,y) in lands:
        #         cv2.circle(image, (x,y),1,(0,0,255),-1)
        # cv2.imshow("Original Image", image)
        # cv2.waitKey(0)        

        fout = open(filename[:-4]+".txt", "wt") #Open a txt file to save the landmarks and the box in the iBUG format
        filename2=np.load(filename[:-4]+'_box.npy') #Load the file with the box coordinates of the top left and bottom right corner
        with open(args["path_location_text"]) as doc: #Open text file in which all information from images will be stored
            for number,line in enumerate(doc):
                if number==0: #Ignore first line since it will be the same as the last
                    continue                 
                if number==1: #In order to keep the format of iBUG dataset in the first line
                    if path[-1]=='/':
                        line='  '+ '<image file='+ '\''+path[path.find('ibug')+5:]+filename +'\''+'>'
                    else:
                        line='  '+ '<image file='+ '\''+path[path.find('ibug')+5:]+'/'+filename +'\''+'>'
                    fout.write(line + '\n')
                if number==2: #Replace box coordinates with those of image
                    nums1=[int(s) for s in line.split("\'") if s.isdigit() or s[1:].isdigit()]
                    line=line.replace(str(nums1[0]),str(filename2[0]))
                    line=line.replace(str(nums1[1]),str(filename2[1]))
                    line=line.replace(str(nums1[2]),str(filename2[2]))
                    line=line.replace(str(nums1[3]),str(filename2[3]))
                    fout.write(line)
                if number>=3 and number<=70: #Replace landmark positions
                    nums=[int(s) for s in line.split("\'") if s.isdigit()]
                    line=line.replace(str(nums[1]),str(lands[number-3][0]))
                    line=line.replace(str(nums[2]),str(lands[number-3][1]))
                    fout.write(line)
                if number>70: #In order to keep the format of iBUG dataset in the last line
                    fout.write(line)
        doc.close()
        fout.close()
        
# Add new images to the training set in the format of the iBUG dataset 

# The folder now has all the images with corresponding 'txt' files in the iBUG format

path2=args["path_location_original_text"] #Get the location of the iBUG original training text file
fout = open("training.txt", "wt") #Create a new file which contains all the training images

count=-1 #As initial index
with open(path2) as f: #Open the original training file and count the total number of lines
    for number,line in enumerate(f):
        count=count+1
f.close()      
 
with open(path2) as f: #Open the original training file and copy its content to the new training file
    for number, line in enumerate(f): #Loop over lines of original training file
        if number==(count-1): #Keep in a variable the prelast line
            prelast=line
        if number==count: #Keep in a variable the last line
            lastline=line
        else: #If not prelast or lastline then copy the content
            fout.write(line)
fout.close() #Close that file in order to save it

#Last line of the above file need to be removed
readFile = open("training.txt") #Open that file again
lines = readFile.readlines() #Read its lines
readFile.close() #Close it
foutfin = open("traininglast.txt",'w') #Create a new file to save the above without the last line
foutfin.writelines([item for item in lines[:-1]]) #Write the lines except the last one

for filename in os.listdir(path): #Loop over the new images that we want to add in the training text file
    if os.path.isfile(os.path.join(path,filename)) and 'npy' in filename: #If the file is a 'npy', ignore it
        continue
    elif os.path.isfile(os.path.join(path,filename)) and 'jpg' in filename: #If the file is a 'jpg', ignore it
        continue
    elif os.path.isfile(os.path.join(path,filename)) and 'png' in filename: #If the file is a 'png', ignore it
        continue
    elif os.path.isfile(os.path.join(path,filename)) and 'training' in filename: #If the file contains 'training', ignore it
        continue
    else:
            print(filename) #Print the filename of the text file
            with open(filename) as g: #Open the text file for a specific image
                for number,line in enumerate(g): #Loop over its lines
                    foutfin.write(line) #Write each line of this file to the final training text files which contains all images
            foutfin.write('\n') #Add a new line after writing each file
            g.close() #Close the text file for the specific image
            
foutfin.write(prelast) #At the end write the 'prelast' line saved above to the final training text file          
foutfin.write(lastline) #At the end write the 'last' line saved above to the final training text file 
foutfin.close() #Close the final training text file in order to save it



