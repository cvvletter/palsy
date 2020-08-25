"""
Created on Sun Jul 19 21:45:22 2020

@author: nsourlos
"""

#Use landmarks to distinguish healthy/patient and then the type of palsy (peripheral or central)

# USAGE
# python classification.py --path-location-images /home/user/Desktop/images/  --path-location-predictor /home/user/Desktop/ --manual-annotation 0 --central-or-peripheral 1

# The path should contain the images (their corresponding face boxes and their landmarks will be estimated).
# The format of them will be 'im01.jpg' or 'im01.png', 'im01.npy' (file with landmarks) and 'im01_box.npy'
# Images can only be 'jpg' or 'png'. The landmark files will be 68*2 for landmarks
# and the box should be an array with 4 elements. 
# Images should have proper names meaning that they should not contain spaces or any strange symbols
# Predictor should be named 'patients_landmarks.dat'
# The last argument is optional. It is used only if we have manual annotation and if we have images of central or
# peripheral palsy we should set its value to 1, otherwise to 0. 

# import the necessary packages
import time
import os
import numpy as np
import cv2
import math
import argparse
import subprocess

start = time.time() #To count running time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--path-location-images", required=True,
 	help="path to folder of images")
ap.add_argument("-p", "--path-location-predictor", required=True,
	help="path to file for facial landmark prediction (landmark predictor)")
ap.add_argument("-o", "--manual-annotation", required=True,
 	help="Did we manually annotate the images (1) or not (0)")
ap.add_argument("-i", "--central-or-peripheral", required=False,
 	help="Are we in a central or peripheral folder (1) or not (0)")
args = vars(ap.parse_args())


def best_fit(X, Y): # Function that finds the best line that finds some points

    xbar = sum(X)/len(X) #Find average value of x coordinates
    ybar = sum(Y)/len(Y) #Find average value of y coordinates
    n = len(X) # or len(Y) #Save in a variable the number of points

    #The two variables below will be used for the estimation of the slope of the line and of its constant
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

#Count how many individuals are classified as having a central palsy, a peripheral palsy or being healthy
countper=0
countcen=0
countheal=0

manual_an=args["manual_annotation"] #Check if we have manual annotation (1) or not (0)
if int(manual_an)==1: #Print message to inform
    print("We have manual annotation") 
else:
    print('We have automatic annotation')
    
pathin=args["path_location_predictor"]
path= args["path_location_images"] #Location of images

for filename in os.listdir(path): #Loop over the folder with images
    fname=filename #Get name of each file in the directory
    #If filename is an 'npy' file or if it is the image with the predicted landmarks or the 'blank' image then continue
    if os.path.isfile(os.path.join(path,filename)) and 'npy' in filename:
        continue
    if os.path.isfile(os.path.join(path,filename)) and 'blank' in filename:
        continue
    if os.path.isfile(os.path.join(path,filename)) and 'land' in filename:
        continue
    elif os.path.isdir(os.path.join(path,fname)): #If the name represents a directory then:
        if int(manual_an)==1:
            continue
        else:
            for filename2 in os.listdir(path+fname): #Loop over files in that directory
                fname2=filename2 #Get filename
                cmd="python palsyfinal.py --shape-predictor patients_landmarks.dat --image "+path+fname+"/"+fname2 #Run landmark detector
                print("Processing Image "+ path+fname+"/"+fname2) #Print file locatio that landmarks are calculated for
                pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                    shell=True, preexec_fn=os.setsid) #Run the above command which normally needs terminal 
                pro.wait() #Wait until execution finish
    else: #If is not a directory (is image) then similar as above:
        if int(manual_an)==1:
            continue
        else:
            cmd="python palsyfinal.py --shape-predictor patients_landmarks.dat --image "+path+fname
            print("Processing Image "+ path+fname)
            pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                shell=True, preexec_fn=os.setsid) 
            pro.wait()

        
os.chdir(path) #Go into that path

#Initialize empty lists for each path that will be filled below
eyeopening=[]
broweyedif=[]
moutheyes=[]
listofnames=[]
eyesnose=[]
eyestopnose=[]
eyestopleftmouth=[]
eyesbottomrightmouth=[]
topnosesidesofmouth=[]
eyeopening0=[]
eyeopening1=[]
eyeopening3=[]
eyeopening5=[]
eyeschinlow=[]
broweyedif1=[]
moutheyes0=[]
moutheyes5=[]
eyeopening3a=[]
eyeopening3b=[]
eyeopeninga=[]
listlefta=[]
listb=[]
listrighta=[]
listlefta1=[]
listrighta1=[]
listlefta2=[]
listrighta2=[]
eyeopeningb=[]
listlefte=[]
listrighte=[]
listline=[]
listlefte1=[]
listrighte1=[]
listlefte2=[]
listrighte2=[]
listlefta3=[]
listrighta3=[]
listlefta4=[]
listrighta4=[]
listlefte3=[]
listrighte3=[]
listlefte4=[]
listrighte4=[]
listeyedif=[]
listbrowdif=[]
listbroweye=[]
listslrightleft=[]
listmouthbrow=[]
totaleyebrowdif=[]
listbroweye2=[]
listangleleft=[]
listangleright=[]
listleftax=[]
listrightax=[]
lenratio=[]

#The following lists are only used for distinguishing palsy type
newmet=[]
newmet2=[]
newmet3=[]
mouriglist=[]
mouleflist=[]

for filename in os.listdir(path): #Loop over files in that path
        if os.path.isfile(os.path.join(path,filename)) and '_box.npy' not in filename: #If it is not the 'npy' box continue
            continue
        elif os.path.isfile(os.path.join(path,filename)) and '_box.npy' in filename:
            facebox=np.load(filename) #Load face box
            image = cv2.imread(filename[:-8]+'.jpg') #Load the image
            lastones='.jpg' #Keep the image type
            if image is None: #If image is not jpg then
                image = cv2.imread(filename[:-8]+'.png') #Load image as png
                lastones='.png' #Keep the image type
            listofnames.append(filename[:-8]+lastones) #Add the name of the image to a list
            lands=np.load(filename[:-8]+'.npy') #Load landmarks

            if int(manual_an)==1:
                if int(args["central_or_peripheral"])==1: #For peripheral or central images do the following:
                    resiz=cv2.resize(image,(900,900)) #Resize image to the same size as when manually annotated
                    fin=image.shape #Get the dimensions of the final image that we want
                    init=resiz.shape #Get the dimensions of the initial image before any transformation
                    lands[:,0]=lands[:,0]*fin[1]/init[1] #Transform the x coordinates of the landmarks
                    lands[:,1]=lands[:,1]*fin[0]/init[0] #Transform the y coordinates of the landmarks
                    lands=lands.astype(int) #Make the landmarks integers since transformation results in float numbers      
         #As a first step the face is cropped and the landmarks are adapted to fit in the size of that crop   
            for i in range(len(facebox)): #Loop over the facebox coordinates
                if facebox[i]<0: #If a coordinate is outside of the image set it to 0
                    facebox[i]=0
            cropped_img = image[facebox[1]:facebox[3],facebox[0]:facebox[2]] #Crop image in x and y direction to include the face only
            
            #The following are just to confirm that the image is cropped properly (It will be)
            # cv2.imshow("Cropped Image", cropped_img)
            # cv2.waitKey(0)
            # cv2.imwrite(filename[:-8]+'crop'+lastones, cropped_img)
                    
            lands[:,0]=lands[:,0]-facebox[0] #Transform the x coordinates of the landmarks based on the cropped image
            lands[:,1]=lands[:,1]-facebox[1] #Transform the y coordinates of the landmarks based on the cropped image

            #The following are just to confirm that the landmarks are transformed properly
            # for (x,y) in lands:
            #     cv2.circle(cropped_img, (x,y),1,(0,0,255),-1)
            # # cv2.imshow("Cropped Image", cropped_img)
            # # cv2.waitKey(0)
            # cv2.imwrite(filename[:-8]+'cropland'+lastones, cropped_img)
            
         #As a second step the cropped face is resized to 900*900 and the landmarks are again transformed to fit in that size  
            init=cropped_img.shape #Get the dimensions of the cropped image
            resi=cv2.resize(cropped_img,(900,900)) #Resize cropped image
            landsnew=lands.copy() #Get a copy of the landmarks
            
            fin=resi.shape #Get the dimensions of the initial image before any transformation
            landsnew[:,0]=landsnew[:,0]*fin[0]/init[0] #Transform the x coordinates of the landmarks based on the cropped image
            landsnew[:,1]=landsnew[:,1]*fin[1]/init[1] #Transform the y coordinates of the landmarks based on the cropped image
            landsnew=landsnew.astype(int) #Make the landmarks integers since transformation results in float numbers
            np.save(filename[:-8]+'_lands',landsnew) #Save the new landmarks
            a=landsnew #Just as a shortcut to be used below
            
            for (x,y) in landsnew:
                cv2.circle(resi, (x,y),1,(0,0,255),-1)
            # cv2.imshow("Cropped Image", cropped_img)
            # cv2.waitKey(0)
            cv2.imwrite(filename[:-8]+'cropres'+lastones, resi)
    
         #As a third step the faces and the corresponding landmarks are rotated so that each face will be aligned based on eyes
            centre_left_eye=np.abs(((a[37]+a[38]+a[40]+a[41]+a[36]+a[39])/6)) #Calculate centre of the left eye (average)
            centre_right_eye=np.abs(((a[43]+a[44]+a[46]+a[47]+a[45]+a[42])/6)) #Calculate centre of the right eye (average)
            dY = centre_left_eye[1] - centre_right_eye[1] #Calculate the y centre of the average of the two eyes
            dX = centre_left_eye[0] - centre_right_eye[0] #Calculate the x centre of the average of the two eyes
            
            #Calculate the angle in which we will rotate face so that eyes are aligned
            angle = np.degrees(np.arctan2(dY, dX)) - 180
            #Compute center (x, y)-coordinates (i.e., the median point) between the two eyes in the input image
            eyesCenter = tuple(i for i in (centre_right_eye+centre_left_eye)/2)
            
            #Create an array of size 68*2 to add new landmarks in it after rotation
            newpoint=np.zeros((len(a),2))
            for i in range(len(a)):
                c=tuple(i for i in a[i]) #Make its landmark position a tuple which is needed as input to the rotation function
                newpoint[i]=rotate(eyesCenter,c,math.radians(-angle)) #Rotate landmarks using function defined above
            #Make new landmark positions integer values since rotation results in float values    
            newpoint=newpoint.astype(int) 
            
            # #The following are only for confirmation        
            # for (x,y) in newpoint:
            #     cv2.circle(resi, (x,y),1,(0,0,255),-1)
            # cv2.imshow("Rotated_Image",resi)
            # cv2.waitKey(0)
    
            # Grab the rotation matrix for rotating and scaling the face which will be used in image rotation
            M = cv2.getRotationMatrix2D(eyesCenter, angle, 1)
    
            sh=resi.shape[0:2] #Get only width and height and not the channels
            output = cv2.warpAffine(resi, M, sh, flags=cv2.INTER_CUBIC) #Apply the matrix transformation to get final image

            for (x,y) in newpoint:
                cv2.circle(output, (x,y),1,(0,0,255),-1)
            cv2.imwrite(filename[:-8]+'last'+lastones, output)  #Save the rotated version of the image along with its landmarks

            ##The following are only for confirmation    
            #     cv2.imshow("po",output)
            #     cv2.waitKey(0) 
            
    ##If 'blank' image is needed then activate the following
    #        height, width, channels = output.shape
    #        blank_image = 255*np.ones((height,width,3), np.uint8)
    #        for (x,y) in newpoint:
    #            cv2.circle(blank_image, (x,y),1,(0,0,255),-1)
    #        cv2.imshow("po",blank_image)
    #        cv2.waitKey(0)
            
#Below Measures are presented
 
   # Measure 1: Distance from centre of eyes to edges of mouth
            #Calculate centre of the right eye (average)
            centre_right_eye=np.abs(((newpoint[43]+newpoint[44]+newpoint[46]+newpoint[47]+newpoint[42]+newpoint[45])/6)) 
            #Calculate centre of the left eye (average)
            centre_left_eye=np.abs(((newpoint[37]+newpoint[38]+newpoint[40]+newpoint[41]+newpoint[36]+newpoint[39])/6)) 
            #Calculate euclidean distance between right eye centre and right edge of mouth
            heightright=np.sqrt(((centre_right_eye[1]-newpoint[54][1])**2)+((centre_right_eye[0]-newpoint[54][0])**2)) 
            #Calculate euclidean distance between left eye centre and left edge of mouth
            heightleft=np.sqrt(((centre_left_eye[1]-newpoint[48][1])**2)+((centre_left_eye[0]-newpoint[48][0])**2)) 
    
            height=np.abs(heightright-heightleft) #Get the absolute difference between the two distances from above
            moutheyes.append(height) #Add that difference for the specific image to a list
            
            
   # Measure 2: Distance from centre of eyes to sides of the nose
            #Calculate euclidean distance between right eye centre and right edge of nose
            heightright2=np.sqrt(((centre_right_eye[1]-newpoint[35][1])**2)+((centre_right_eye[0]-newpoint[35][0])**2)) 
            #Calculate euclidean distance between left eye centre and left edge of nose  
            heightleft2=np.sqrt(((centre_left_eye[1]-newpoint[31][1])**2)+((centre_left_eye[0]-newpoint[31][0])**2)) 
            
            height2=np.abs(heightright2-heightleft2) #Get the absolute difference between the two distances from above
            eyesnose.append(height2) #Add that difference for the specific image to a list
            
            
   # Measure 3: Euclidean Distance of each eye from the top of the nose       
            #Calculate euclidean distance between right eye centre and top of nose
            heightright3=np.sqrt(((centre_right_eye[1]-newpoint[30][1])**2)+((centre_right_eye[0]-newpoint[30][0])**2)) 
            #Calculate euclidean distance between left eye centre and top of nose
            heightleft3=np.sqrt(((centre_left_eye[1]-newpoint[30][1])**2)+((centre_left_eye[0]-newpoint[30][0])**2)) 
            
            height3=np.abs(heightright3-heightleft3) #Get the absolute difference between the two distances from above
            eyestopnose.append(height3) #Add that difference for the specific image to a list
            
            
   # Measure 4: Euclidean Distance between each eye and the top left corner of the mouth
            topav=(newpoint[50]+newpoint[51]+newpoint[52])/3 #Average  of top left corner of the mouth
            #Calculate euclidean distance between right eye centre and average of the left corner of mouth
            heightright4=np.sqrt(((centre_right_eye[1]-topav[1])**2)+((centre_right_eye[0]-topav[0])**2)) 
            #Calculate euclidean distance between left eye centre and average of the left corner of mouth
            heightleft4=np.sqrt(((centre_left_eye[1]-topav[1])**2)+((centre_left_eye[0]-topav[0])**2)) 
            
            height4=np.abs(heightright4-heightleft4) #Get the absolute difference between the two distances from above
            eyestopleftmouth.append(height4) #Add that difference for the specific image to a list
            
            
   # Measure 5: Distance between eyes and bottom corner of the right side of the mouth     
            #Find the lowest point in terms of y coordinates
            if newpoint[56][1]<=newpoint[57][1] and newpoint[56][1]<=newpoint[58][1]:
                rightbotmouth=newpoint[56]
            elif newpoint[57][1]<=newpoint[56][1] and newpoint[57][1]<=newpoint[58][1]:
                rightbotmouth=newpoint[57]
            else:
                rightbotmouth=newpoint[58]
                
            #Calculate euclidean distance between right eye centre and lowest point on the right side of mouth
            heightright5=np.sqrt(((centre_right_eye[1]-rightbotmouth[1])**2)+((centre_right_eye[0]-rightbotmouth[0])**2)) 
            #Calculate euclidean distance between left eye centre and lowest point on the right side of mouth 
            heightleft5=np.sqrt(((centre_left_eye[1]-rightbotmouth[1])**2)+((centre_left_eye[0]-rightbotmouth[0])**2))  
            
            height5=np.abs(heightright5-heightleft5) #Get the absolute difference between the two distances from above
            eyesbottomrightmouth.append(height5) #Add that difference for the specific image to a list
            
            
   # Measure 6: Difference in distance between top of nose and each side of the mouth   
            #Calculate euclidean distance between top of nose and right side of the mouth
            heightright6=np.sqrt(((newpoint[30][1]-newpoint[54][1])**2)+((newpoint[30][0]-newpoint[54][0])**2)) 
            #Calculate euclidean distance between top of nose and left side of the mouth
            heightleft6=np.sqrt(((newpoint[30][1]-newpoint[48][1])**2)+((newpoint[30][0]-newpoint[48][0])**2)) 
    
            height6=np.abs(heightright6-heightleft6) #Get the absolute difference between the two distances from above
            topnosesidesofmouth.append(height6) #Add that difference for the specific image to a list
            
   
    # Measure 7: Euclidean Distance between each eye and lowest point of the chin
            #Find lowest point of chin in terms of y coordinates
            if newpoint[7][1]>=newpoint[8][1] and newpoint[7][1]>=newpoint[9][1]:
                chinlow=newpoint[7]
            elif newpoint[8][1]>=newpoint[7][1] and newpoint[8][1]>=newpoint[9][1]:
                chinlow=newpoint[8]
            else:
                chinlow=newpoint[9]
                
            #Calculate euclidean distance between centre of right eye and lowest point of the chin    
            heightright7=np.sqrt(((centre_right_eye[1]-chinlow[1])**2)+((centre_right_eye[0]-chinlow[0])**2)) 
            #Calculate euclidean distance between centre of left eye and lowest point of the chin
            heightleft7=np.sqrt(((centre_left_eye[1]-chinlow[1])**2)+((centre_left_eye[0]-chinlow[0])**2)) 
            
            height7=np.abs(heightright7-heightleft7) #Get the absolute difference between the two distances from above
            eyeschinlow.append(height7) #Add that difference for the specific image to a list

            
   # Measure 8, 53: Calculate difference in amount of eyelid opening between eyes (1)
            #Find highest point of left eye in terms of y coordinates
            if newpoint[37][1]>newpoint[38][1]:
                lefteye1=newpoint[37]
            else:
                lefteye1=newpoint[38]
            #Find lowest point of left eye in terms of y coordinates
            if newpoint[40][1]<newpoint[41][1]:
                lefteye2=newpoint[40]
            else:
                lefteye2=newpoint[41]
                
            lefteye=np.sqrt(((lefteye1[1]-lefteye2[1])**2)) #Find height of left eye opening
            
            #Find highest point of right eye in terms of y coordinates
            if newpoint[43][1]>newpoint[44][1]:
                righteye1=newpoint[43]
            else:
                righteye1=newpoint[44]
            
            #Find lowest point of right eye in terms of y coordinates
            if newpoint[46][1]<newpoint[47][1]:
                righteye2=newpoint[46]
            else:
                righteye2=newpoint[47]
                
            righteye=np.sqrt(((righteye1[1]-righteye2[1])**2)) #Find height of right eye opening

            opening=np.abs(lefteye-righteye) #Get absolute values of difference in amount of eyelid opening between eyes
            eyeopening.append(100*opening) #Add that difference for the specific image to a list (*100 for better scaling)
            rightlen=np.sqrt((newpoint[36][0]-newpoint[39][0])**2) #Calculate eye length for right eye
            leftlen=np.sqrt((newpoint[42][0]-newpoint[45][0])**2) #Calculate eye length for left eye
            rightratio=righteye/rightlen #Calculate ratio for right eye
            leftratio=lefteye/leftlen #Calculate ratio for left eye
            lenratio.append(100*rightratio/leftratio) #Calculate ratio of the two eye ratios
            
   # Measures 9 and 10: A second and third way to calculate amount of eyelid opening
            #Calculate difference in y coordinates for middle-left points of left eye
            lefteye3=np.abs(newpoint[37][1]-newpoint[41][1]) 
            #Calculate difference in y coordinates for middle-right points of left eye
            lefteye4=np.abs(newpoint[38][1]-newpoint[40][1]) 
            #Calculate difference in y coordinates for middle-left points of right eye
            righteye3=np.abs(newpoint[43][1]-newpoint[47][1]) 
            #Calculate difference in y coordinates for middle-right points of right eye
            righteye4=np.abs(newpoint[44][1]-newpoint[46][1]) 
            
            eyeopeninga.append(100*np.abs(lefteye3+lefteye4-righteye3-righteye4)) #Add that difference for the specific image 
            # to a list multiplied by 100 for better scaling and with a few additions and subtractions between these values
            eyeopeningb.append(np.abs((newpoint[39][0]-newpoint[36][0])-(newpoint[45][0]-newpoint[42][0]))) #Add to a list
            #This is the difference between the opening in x direction of each eye

            
   # Measure 11: Difference between the difference of the distance of the lowest point of each eye to the highest of the corresponding brow
            #Find the highest point of the rightbrow
            if newpoint[23][1]>=newpoint[24][1] and newpoint[23][1]>=newpoint[25][1]:
                rightbrow=newpoint[23]
            elif newpoint[24][1]>=newpoint[23][1] and newpoint[24][1]>=newpoint[25][1]:
                rightbrow=newpoint[24]
            else:
                rightbrow=newpoint[25]
            #Find the highest point of the leftbrow
            if newpoint[18][1]>=newpoint[19][1] and newpoint[18][1]>=newpoint[20][1]:
                leftbrow=newpoint[18]
            elif newpoint[19][1]>=newpoint[20][1] and newpoint[19][1]>=newpoint[18][1]:
                leftbrow=newpoint[19]
            else:
                leftbrow=newpoint[20]

            euclright=np.sqrt(((righteye2[1]-rightbrow[1])**2)) #Find the height difference between right eye and right brow
            eucleft=np.sqrt(((lefteye2[1]-leftbrow[1])**2)) #Find the height difference between left eye and left brow
            
            broweyedif.append(100*np.abs(euclright-eucleft)) #Add that difference for the specific image to a list


   # Measure 12: Difference of distance between tip of nose and each brow
            euclright2=np.sqrt(((newpoint[30][1]-rightbrow[1])**2)) #Distance from tip of nose to right brow
            eucleft2=np.sqrt(((newpoint[30][1]-leftbrow[1])**2)) #Distance from tip of nose to left brow
            eyeopening0.append(100*np.abs(euclright2-eucleft2)) #Add that difference for the specific image to a list
   # Measure 13: Difference between most-left point of nose and each brow 
            euclright3=np.sqrt(((newpoint[35][1]-rightbrow[1])**2)) #Distance of most-left point on nose to right brow
            eucleft3=np.sqrt(((newpoint[31][1]-leftbrow[1])**2)) #Distance of most-left point on nose to right brow
            eyeopening1.append(100*np.abs(euclright3-eucleft3)) #Add that difference for the specific image to a list
            

   # Measure 14: Distance between sides of mouth and corresponding brows
            #Calculate highest point on the right side of mouth
            if newpoint[54][1]>=newpoint[53][1] and newpoint[54][1]>=newpoint[55][1]:
                mourig=newpoint[54]
            elif newpoint[53][1]>=newpoint[54][1] and newpoint[53][1]>=newpoint[55][1]:
                mourig=newpoint[53]
            else:
                mourig=newpoint[55]
            #Calculate highest point on the left side of mouth
            if newpoint[48][1]>=newpoint[59][1] and newpoint[48][1]>=newpoint[49][1]:
                moulef=newpoint[48]
            elif newpoint[49][1]>=newpoint[48][1] and newpoint[49][1]>=newpoint[59][1]:
                moulef=newpoint[49]
            else:
                moulef=newpoint[59]
            
            euclright4=np.sqrt(((mourig[1]-rightbrow[1])**2)) #Distance of right side of mouth to right brow
            eucleft4=np.sqrt(((moulef[1]-leftbrow[1])**2)) #Distance of left side of mouth to left brow
            
            eyeopening3.append(100*np.abs(euclright4-eucleft4)) #Add that difference for the specific image to a list
            #Multiplied by 100 for scaling
            
            
   # Measure 15: Distance between sides of mouth and corresponding brows (try to amplify it)
            #Calculate highest point on the right side of mouth
            if newpoint[54][1]>=newpoint[53][1] and newpoint[54][1]>=newpoint[55][1]:
                mourig2=newpoint[54]
            elif newpoint[53][1]>=newpoint[54][1] and newpoint[53][1]>=newpoint[55][1]:
                mourig2=newpoint[53]
            else:
                mourig2=newpoint[55]
            #Calculate lowest point on the left side of mouth
            if newpoint[48][1]<=newpoint[59][1] and newpoint[48][1]<=newpoint[49][1]:
                moulef2=newpoint[48]
            elif newpoint[49][1]<=newpoint[48][1] and newpoint[49][1]<=newpoint[59][1]:
                moulef2=newpoint[49]
            else:
                moulef2=newpoint[59]
            
            euclright4a=np.sqrt(((mourig2[1]-rightbrow[1])**2)) #Distance of right side of mouth to right brow
            eucleft4a=np.sqrt(((moulef2[1]-leftbrow[1])**2)) #Distance of left side of mouth to left brow

            eyeopening3a.append(100*np.abs(euclright4a-eucleft4a)) #Add that difference for the specific image to a list
            #Multiplied by 100 for scaling
            
            
   # Measure 16: Distance between sides of mouth and corresponding brows (try to amplify it)
            #Calculate lowest point on the right side of mouth     
            if newpoint[54][1]<=newpoint[53][1] and newpoint[54][1]<=newpoint[55][1]:
                mourig3=newpoint[54]
            elif newpoint[53][1]<=newpoint[54][1] and newpoint[53][1]<=newpoint[55][1]:
                mourig3=newpoint[53]
            else:
                mourig3=newpoint[55]
            #Calculate highest point on the left side of mouth
            if newpoint[48][1]>=newpoint[59][1] and newpoint[48][1]>=newpoint[49][1]:
                moulef3=newpoint[48]
            elif newpoint[49][1]>=newpoint[48][1] and newpoint[49][1]>=newpoint[59][1]:
                moulef3=newpoint[49]
            else:
                moulef3=newpoint[59]
            
            euclright4b=np.sqrt(((mourig3[1]-rightbrow[1])**2)) #Distance of right side of mouth to right brow
            eucleft4b=np.sqrt(((moulef3[1]-leftbrow[1])**2)) #Distance of left side of mouth to left brow

            eyeopening3b.append(100*np.abs(euclright4b-eucleft4b)) #Add that difference for the specific image to a list
            #Multiplied by 100 for scaling


   # # Measure 17: Distance between lowest point on chin and brows
   #          euclright5=np.sqrt(((chinlow[1]-rightbrow[1])**2)) #Distance of lowest point of chin to right brow
   #          eucleft5=np.sqrt(((chinlow[1]-leftbrow[1])**2)) #Distance of lowest point of chin to left brow
            
   #          eyeopening4.append(100*np.abs(euclright5-eucleft5)) #Add that difference for the specific image to a list
            
            
   # # Measure 18: Difference of eyebrow heights      
   #          brows=np.sqrt(((rightbrow[1]-leftbrow[1])**2)) #Difference of eyebrow heights
            
   #          eyeopening6.append(100*brows) #Add that difference for the specific image to a list (multiplied by 100 for scaling)
            
            
   # Measure 19: Distance between lowest point on chin and almost tip of nose         
            height8=np.sqrt(((chinlow[1]-newpoint[30][1])**2))
            broweyedif1.append(10*height8) #Add that difference for the specific image to a list
     
   # Measures 20: Differences of the average of 3 left from the average of 3 right points on the top of each eye
            left1=(newpoint[37]+newpoint[38]+newpoint[39])/3
            left2=(newpoint[36]+newpoint[40]+newpoint[41])/3
            right1=(newpoint[43]+newpoint[44]+newpoint[45])/3
            right2=(newpoint[42]+newpoint[46]+newpoint[47])/3
            
            left3=np.abs(np.sqrt(((left1[1]-left2[1])**2)))
            right3=np.abs(np.sqrt(((right1[1]-right2[1])**2)))

            height9=np.abs(right3-left3) #Absolute difference of differences of the two eyes
            moutheyes0.append(100*height9) #Add that difference for the specific image to a list (*100 for scaling)
            
            
   # Measure 22: Calculate difference of opening of each side of the mouth  
            moutheyes5.append(100*np.abs(mourig[1]-moulef[1])) #Add that difference for the specific image to a list
       
        
   # Measures 23, 24, 25: Fit the best lines on left and right brows
            x_left=[newpoint[17][0],newpoint[18][0],newpoint[19][0],newpoint[20][0],newpoint[21][0]]
            y_left=[newpoint[17][1],newpoint[18][1],newpoint[19][1],newpoint[20][1],newpoint[21][1]]
            x_right=[newpoint[22][0],newpoint[23][0],newpoint[24][0],newpoint[25][0],newpoint[26][0]]
            y_right=[newpoint[22][1],newpoint[23][1],newpoint[24][1],newpoint[25][1],newpoint[26][1]]
            a,b=best_fit(x_left,y_left)
            listlefta.append(a) #Add that difference for the specific image to a list
            a1,b1=best_fit(x_right,y_right)
            listrighta.append(a1) #Add that difference for the specific image to a list
            
            #Add the difference of the constant values of these lines to a list
           # listb.append(100*np.abs(b1-b)) #Add that difference for the specific image to a list
            
   # Measure 26: Calculate difference of top points in each brow and eye from each other
            lefteyeall=[newpoint[36][1], newpoint[37][1], newpoint[38][1], newpoint[39][1], newpoint[40][1], newpoint[41][1]]
            righteyeall=[newpoint[42][1], newpoint[43][1], newpoint[44][1],newpoint[45][1],newpoint[46][1],newpoint[47][1]]
            
            totaleyebrowdif.append(np.abs(np.abs(np.max(y_left)-np.max(lefteyeall))-np.abs(np.max(y_right)-np.max(righteyeall)))) 

            
    # Measures 27 and 28: Best line fit of 3 most left points of each brow and save the slope of each brow to a list
            x_left1=[newpoint[17][0],newpoint[18][0],newpoint[19][0]]
            y_left1=[newpoint[17][1],newpoint[18][1],newpoint[19][1]]
            x_right1=[newpoint[22][0],newpoint[23][0],newpoint[24][0]]
            y_right1=[newpoint[22][1],newpoint[23][1],newpoint[24][1]]
            a2,b2=best_fit(x_left1,y_left1)
            listlefta1.append(a2) #Add that difference for the specific image to a list
            a3,b3=best_fit(x_right1,y_right1)
            listrighta1.append(a3) #Add that difference for the specific image to a list
            # listbrowslopedif.append(a3-a1)
        
        
    # Measures 29 and 30: Best line fit of 3 middle points of each brow and keep the slope of each one of them to a list
            x_left2=[newpoint[19][0],newpoint[20][0],newpoint[21][0]]
            y_left2=[newpoint[19][1],newpoint[20][1],newpoint[21][1]]
            x_right2=[newpoint[24][0],newpoint[25][0],newpoint[26][0]]
            y_right2=[newpoint[24][1],newpoint[25][1],newpoint[26][1]]
            a4,b4=best_fit(x_left2,y_left2)
            listlefta2.append(a4) #Add that difference for the specific image to a list
            a5,b5=best_fit(x_right2,y_right2)
            listrighta2.append(a5) #Add that difference for the specific image to a list
        
        
   # Measures 31 and 32: Best line fit of 3 most left points of each eye and keep the slope of each eye to a list
            x_lefteye=[newpoint[36][0],newpoint[37][0],newpoint[38][0]]
            y_lefteye=[newpoint[36][1],newpoint[37][1],newpoint[38][1]]
            x_righteye=[newpoint[42][0],newpoint[43][0],newpoint[44][0]]
            y_righteye=[newpoint[42][1],newpoint[43][1],newpoint[44][1]]
            ae,be=best_fit(x_lefteye,y_lefteye)
            listlefte.append(ae) #Add that difference for the specific image to a list
            ae1,be1=best_fit(x_righteye,y_righteye)
            listrighte.append(ae1) #Add that difference for the specific image to a list
            
            
    # Measure 34: Distance from top of nose to the farthest point of each eye 
            listline.append(np.abs((newpoint[27][0]-newpoint[36][0])-(newpoint[27][0]-newpoint[45][0]))) #Add that to a list
            
            
    # Measures 35, 36 and 37: Best line fit of 2 most left points of each eye and keep slope and difference of slopes to a list
            x_lefteye1=[newpoint[36][0],newpoint[37][0]]
            y_lefteye1=[newpoint[36][1],newpoint[37][1]]
            x_righteye1=[newpoint[42][0],newpoint[43][0]]
            y_righteye1=[newpoint[42][1],newpoint[43][1]]
            ae2,be2=best_fit(x_lefteye1,y_lefteye1)
            listlefte1.append(ae2) #Add that difference for the specific image to a list
            ae3,be3=best_fit(x_righteye1,y_righteye1)
            listrighte1.append(ae3) #Add that difference for the specific image to a list
            listslrightleft.append(np.abs(ae3-ae2))
        
    # Measures 38 and 39: Best line fit of the 2 most right points of each eye and keep slope in a list
            x_lefteye2=[newpoint[38][0],newpoint[39][0]]
            y_lefteye2=[newpoint[38][1],newpoint[39][1]]
            x_righteye2=[newpoint[44][0],newpoint[45][0]]
            y_righteye2=[newpoint[44][1],newpoint[45][1]]
            ae4,be4=best_fit(x_lefteye2,y_lefteye2)
            listlefte2.append(ae4) #Add that difference for the specific image to a list
            ae5,be5=best_fit(x_righteye2,y_righteye2)
            listrighte2.append(ae5) #Add that difference for the specific image to a list
            
            
    # Measures 40 and 41: Best line fit of the 2 most left points of each brow and keep slopes in a list
            x_left3=[newpoint[17][0],newpoint[18][0]]
            y_left3=[newpoint[17][1],newpoint[18][1]]
            x_right3=[newpoint[22][0],newpoint[23][0]]
            y_right3=[newpoint[22][1],newpoint[23][1]]
            a6,b6=best_fit(x_left3,y_left3)
            listlefta3.append(a6) #Add that difference for the specific image to a list
            a7,b7=best_fit(x_right3,y_right3)
            listrighta3.append(a7) #Add that difference for the specific image to a list
        
        
    # Measures 42 and 43: Best line fit of the 2 most right points of each brow and keep slopes in a list
            x_left4=[newpoint[20][0],newpoint[21][0]]
            y_left4=[newpoint[20][1],newpoint[21][1]]
            x_right4=[newpoint[25][0],newpoint[26][0]]
            y_right4=[newpoint[25][1],newpoint[26][1]]
            a8,b8=best_fit(x_left4,y_left4)
            listlefta4.append(a8) #Add that difference for the specific image to a list
            a9,b9=best_fit(x_right4,y_right4)
            listrighta4.append(a9) #Add that difference for the specific image to a list
         
            
    # Measures 44 and 45: Best line fit of the 2 most left bottom points of each eye and keep slopes
            x_lefteye3=[newpoint[36][0],newpoint[41][0]]
            y_lefteye3=[newpoint[36][1],newpoint[41][1]]
            x_righteye3=[newpoint[42][0],newpoint[47][0]]
            y_righteye3=[newpoint[42][1],newpoint[47][1]]
            ae6,be6=best_fit(x_lefteye3,y_lefteye3)
            listlefte3.append(ae6) #Add that difference for the specific image to a list
            ae7,be7=best_fit(x_righteye3,y_righteye3)
            listrighte3.append(ae7) #Add that difference for the specific image to a list
        
        
    # Measures 46 and 47: Best line fit of the 2 most right bottom points of each eye and keep slopes
            x_lefteye4=[newpoint[39][0],newpoint[40][0]]
            y_lefteye4=[newpoint[39][1],newpoint[40][1]]
            x_righteye4=[newpoint[45][0],newpoint[46][0]]
            y_righteye4=[newpoint[45][1],newpoint[46][1]]
            ae8,be8=best_fit(x_lefteye4,y_lefteye4)
            listlefte4.append(ae8) #Add that difference for the specific image to a list
            ae9,be9=best_fit(x_righteye4,y_righteye4)
            listrighte4.append(ae9) #Add that difference for the specific image to a list
         
            
    # Measure 48: Difference of the highest point of both eyes from the lowest of both eyes
            listeyehighlow=[newpoint[36][1], newpoint[37][1], newpoint[38][1], newpoint[39][1], newpoint[40][1], 
                            newpoint[41][1], newpoint[42][1], newpoint[43][1], newpoint[44][1],newpoint[45][1],
                            newpoint[46][1],newpoint[47][1]]
            eyehigh=np.max(listeyehighlow)
            eyelow=np.min(listeyehighlow)
            listeyedif.append(np.abs(eyehigh-eyelow)) #Add that difference for the specific image to a list
          
            
    # Measure 49, 50, 51: Difference of the highest point of both brows from the lowest of both brows
            listbrowhighlow=[newpoint[17][1], newpoint[18][1],newpoint[19][1],newpoint[20][1],newpoint[21][1],
                             newpoint[22][1],newpoint[23][1],newpoint[24][1], newpoint[25][1],newpoint[26][1] ]
            browhigh=np.max(listbrowhighlow)
            browlow=np.min(listbrowhighlow)
            listbrowdif.append(np.abs(browhigh-browlow)) #Add that difference for the specific image to a list
            #Difference of the highest point of both brows from the lowest of both eyes
            listbroweye.append(np.abs(browhigh-eyelow)) #Add that difference for the specific image to a list
            #Difference of the highest point of both brows from the highest of both eyes
            listbroweye2.append(np.abs(browhigh-eyehigh)) #Add that difference for the specific image to a list
            
            
    # Measure 52: Difference of the highest from the lowest point of the 4 most-left and 4 most-right points of the mouth
            listmouthhighlow=[newpoint[48][1],newpoint[49][1],newpoint[59][1],newpoint[60][1],newpoint[53][1],
                              newpoint[54][1],newpoint[55][1],newpoint[64][1]]
            mouthlow=np.min(listmouthhighlow)
            mouthhigh=np.max(listmouthhighlow)
            listmouthbrow.append(np.abs(mouthlow-browhigh)) #Add that difference for the specific image to a list
            
            
    #The following metrics might be used for manual metrics of patients
            rightside=np.abs(mourig3[1]-righteye2[1]) #Distance of right side of the mouth to right eye
            leftside=np.abs(moulef2[1]-lefteye2[1]) #Distance of left side of the mouth to left eye
            
            difrightlefteye=righteye-lefteye #Difference of left and right eye openings
            difrightleftbrow=leftbrow[1]-rightbrow[1] #Difference of left and right eyebrow heights
            newmet.append(difrightlefteye) #Append the value to a list
            newmet2.append(difrightleftbrow) #Append the value to a list
            mouriglist.append(mourig3[1]) #Append lowest point of right eye to a list
            mouleflist.append(moulef2[1]) #Append lowest point of left eye to a list
  
     
            #Initialize empty lists to be used below    
            eyeopening2=[]
            broweyedif2=[]
            
            for i in range(len(eyeopening)): #Loop over the number of images in a specific path
                #The folowwing are used to distinguish palsy type
            
                #Sum all distances from the eye to mouth, nose, chin etc. 
                eyeopening2.append(moutheyes[i]+eyesnose[i]+eyestopnose[i]+eyestopleftmouth[i]+
                                   eyesbottomrightmouth[i]+topnosesidesofmouth[i]+eyeschinlow[i])
                #Sum all distances from the brow to mouth, nose, chin etc.
                broweyedif2.append(eyeopening1[i]+eyeopening3[i]+eyeopening0[i]+moutheyes5[i]) #Add that difference to a list
                
            if int(manual_an)==1: #These are the threshods for manual annotation
                       if (eyeopening[i]>1900 or broweyedif[i]>1800 or eyeopening0[i]>1700 or eyeopening1[i]>2400 or eyeopening3[i]>3000 or moutheyes5[i]>2300 or eyeopening3b[i]<6000 or listlefte4[i]>490 or eyeopeninga[i]>3000 or listrighta[i]<-100 or listrighte3[i]<40) or totaleyebrowdif[i]>32 or listrighte[i]<200   :   
                       #Above are the best thresholds found 
                           print("Image {} is classified as patient (manual annotation)".format(listofnames[i]))
                           if (eyeopening[i]>2700 or broweyedif[i]>4500 or eyeopening0[i]>4000 or eyeopening1[i]>8000 or broweyedif1[i]>6500 or broweyedif1[i]<3300 or eyeopening3[i]<380 or moutheyes0[i]>2000  or eyeopeninga[i]>5500 ) or (moutheyes[i]>=120 or eyesnose[i]>=60 or eyestopleftmouth[i]>=80 or topnosesidesofmouth[i]>=110 or eyesbottomrightmouth[i]>100) or eyeopening2[i]>630 or listline[i]>700 or listlefte4[i]>570 or listbroweye[i]>85 or eyeopening3a[i]>15500 or  listrighta2[i]>-10 or listrighte2[i]>500 or listrighte3[i]>250  or broweyedif2[i]<2000 or listrighta4[i]<-1600 or listbrowdif[i]>140 or lenratio[i]<20 :
                                print("Patient is classified as having a peripheral palsy")
                                print("\n")
                                countper+=1
                           else:
                                print("Patient is classified as having a central palsy")
                                print("\n")
                                countcen+=1
                       else: 
                           print("Image {} is classified as healthy (manual annotation)".format(listofnames[i]))
                           print("\n")
                           countheal+=1
            elif int(manual_an)==0: #These are the threshods for automatic annotation
                       if (moutheyes[i]>30 or eyestopnose[i]>110 or eyeschinlow[i]>35 or mouriglist[i]>700 or broweyedif[i]>1800 or eyeopeninga[i]>3000 or listline[i]>650 or listbroweye2[i]<35 or broweyedif1[i]<3200 or listlefte4[i]<200 or listrighta1[i]<110 or listlefta2[i]>150 or eyeopening1[i]>2400 or eyeopening3[i]>2900 or moutheyes5[i]>2200 or eyeopening3b[i]<6300 or listline[i]<450 or listmouthbrow[i]>520 or listlefte[i]<200 or listbrowdif[i]>140 or eyesbottomrightmouth[i]>85 or eyesbottomrightmouth[i]<15  ):
                           print("Image {} is classified as patient (automatic annotation)".format(listofnames[i]))
                           if (eyeopening[i]>1400 or eyeopening1[i]>4100 or eyeopening3[i]>5900 or broweyedif1[i]>6600 or broweyedif1[i]<3300 or moutheyes5[i]>4200 or eyeopeninga[i]>2600 or listline[i]>650 or eyeopening3a[i]>15500 or listlefte2[i]<-115 or broweyedif[i]>3200 or broweyedif2[i]>15500 or listlefte4[i]>460 or listbroweye2[i]>120 or listlefta1[i]<100 or listrighte3[i]>270 or eyeopeningb[i]>23 or eyeopening0[i]>2600 or listslrightleft[i]>490 or listmouthbrow[i]>550 or moutheyes0[i]>1980 or listlefta2[i]<-47 ):
                           #if (eyeopening[i]>1000 or eyeopening1[i]>3800 or eyeopening3[i]>5500 or broweyedif1[i]>6600 or broweyedif1[i]<3300 or moutheyes5[i]>3900 or eyeopeninga[i]>1600 or listline[i]>650 or eyeopening3a[i]>15000 or listlefte2[i]<-80 or broweyedif[i]>2800 or broweyedif2[i]>14000 or broweyedif2[i]<1000 or listlefte4[i]>440 or listbroweye[i]>85 or listbroweye2[i]>130 or listlefta1[i]<110 or listrighta2[i]>-50 or listrighte3[i]>270 or eyeopeningb[i]>23):
                                print("Patient is classified as having a peripheral palsy")
                                print("\n")
                                countper+=1
                           else:
                                print("Patient is classified as having a central palsy")
                                print("\n")
                                countcen+=1
                       else:
                           print("Image {} is classified as healthy (automatic annotation)".format(listofnames[i]))
                           print("\n")
                           countheal+=1

end=time.time()
print("Total running time was {} seconds".format(end-start))
print("From a total of {} images, {} are classified as healthy, {} as having a central and {} as having a peripheral palsy".format(countper+countcen+countheal, countheal,countcen,countper))