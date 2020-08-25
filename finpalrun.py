"""
Created on Wed Jul  1 19:16:36 2020
@author: nsourlos
"""

#Usage: "python finpalrun.py --path-location DIRECTORY_WITH_SHAPE_PREDICTOR --image-folder DIRECTORY_WITH_IMAGES"
#An example is shown below:
#Usage: "python finpalrun.py --path-location /home/user/Desktop/custom_shape_predictor --image-folder /home/user/Desktop/images/" 
#The image folder should contain only images and/or directories! Not any other kind of file

#Import packages
import os
import subprocess
import argparse
import time

#Just to calculate total execution time
beg=time.time()

#Get the arguments
ap1 = argparse.ArgumentParser()
ap1.add_argument("-l", "--path-location", required=True,
	help="path to file for facial landmark prediction")
ap1.add_argument("-i", "--image-folder", required=True,
	help="path to folder with input images")
args1 = vars(ap1.parse_args())

#Paths to be used from arguments
pathin=args1["path_location"]
path2=args1["image_folder"]
 
for filename in os.listdir(path2): #Loop over the folder with images
    fname=filename #Get name of each file in the directory
    #If filename is an 'npy' file or if it is the image with the predicted landmarks or the 'blank' image then continue
    if os.path.isfile(os.path.join(path2,filename)) and 'npy' in filename:
        continue
    if os.path.isfile(os.path.join(path2,filename)) and 'blank' in filename:
        continue
    if os.path.isfile(os.path.join(path2,filename)) and 'land' in filename:
        continue
    elif os.path.isdir(os.path.join(path2,fname)): #If the name represents a directory then:
        for filename2 in os.listdir(path2+fname): #Loop over files in that directory
            fname2=filename2 #Get filename
            cmd="python palsyfinal.py --shape-predictor patients_landmarks.dat --image "+path2+fname+"/"+fname2 #Run landmark detector
            print("Processing Image "+ path2+fname+"/"+fname2) #Print file locatio that landmarks are calculated for
            pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                shell=True, preexec_fn=os.setsid) #Run the above command which normally needs terminal 
            pro.wait() #Wait until execution finish
    else: #If is not a directory (is image) then similar as above:
        cmd="python palsyfinal.py --shape-predictor patients_landmarks.dat --image "+path2+fname
        print("Processing Image "+ path2+fname)
        pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                            shell=True, preexec_fn=os.setsid) 
        pro.wait()
        
end=time.time()
print(end-beg)