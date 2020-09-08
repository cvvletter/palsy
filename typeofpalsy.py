"""
Created on Tue Jul  7 12:35:24 2020

@author: nsourlos
"""

#Use landmarks to distinguish healthy/patient and then the type of palsy (peipheral or central)

# USAGE
# python typeofpalsy.py --path-location-peripheral-images /home/user/Desktop/perimages  --path-location-central-images /home/user/Desktop/cenimages --path-location-healthy-images /home/user/Desktop/healimages --manual-annotation 1 --check-patient 1

# Each image folder should contain the images along with their corresponding face boxes and their landmarks.
# The format of them should be 'im01.jpg' or 'im01.png', 'im01.npy' (file with landmarks) and 'im01_box.npy'
# Images can only be 'jpg' or 'png'. The landmark files should be 68*2 for landmarks
# and the box should be an array with 4 elements. The last two argument indicate if we have manual annotation (1) or not (0)
# and if we want to check metrics for patients (1) or not (0)

# import the necessary packages
import time
import os
import numpy as np
import cv2
import math
import argparse
import matplotlib.pyplot as plt

start = time.time() #To count running time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--path-location-peripheral-images", required=True,
 	help="path to folder of images with peripheral palsy")
ap.add_argument("-p", "--path-location-central-images", required=True,
 	help="path to folder of images with central palsy")
ap.add_argument("-k", "--path-location-healthy-images", required=True,
 	help="path to folder of images with healthy individuals")
ap.add_argument("-o", "--manual-annotation", required=True,
 	help="Did we manually annotate the images (1) or not (0)")
ap.add_argument("-q", "--check-patient", required=True,
 	help="Do we want to check metrics for patients (1) or not (0)")
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


manual_an=args["manual_annotation"] #Check if we have manual annotation (1) or not (0)
if int(manual_an)==1: #Print message to inform
    print("We have manual annotation") 
else:
    print('We have automatic annotation')

pathper= args["path_location_peripheral_images"] #Location of peripheral images
pathcen= args["path_location_central_images"] #Location of central images
pathheal= args["path_location_healthy_images"] #Location of healthy images
pathnew=[pathper,pathcen,pathheal] #Combine all paths to a list

#Initialize empty lists of false positives, false negatives, true positives and true negatives
#as well as of the total trues and total false combined
fplis=[]
fnlis=[]
tplis=[]
tnlis=[]
combtrue=[]
combfalse=[]

#These are the same but to distinguish central/peripheral palsy
fplis1=[]
fnlis1=[]
tplis1=[]
tnlis1=[]

#The 2 lists below are used because in True Positives and False Negatives in our code, both central and peripheral palsy are counted. 
#Each time we loop only in one of these folders and so, only one value is appended at a time. We combine them in order to get the 
#final number of tp and fn in the lists below
tplast=[]
fnlast=[]

#Same for patients
tplast1=[]
fnlast1=[]

j=1 #This index was used during the grid search in order to find the best values for the threshold. It was used as a loop index
#in each of the metrics below. Now it can just be initialized to 1. 

#Below are for manual annotation
pointstoplot=[]

#Original metrics resulted from the procedure of Algorithm 2
# metrnames=['eyeopening[i]>1950.0', 'broweyedif[i]>2900.0', 'moutheyes[i]>59.93162391836316', 'eyesnose[i]>45.86578780511698', 
#             'eyestopnose[i]>107.1074562840437', 'eyestopleftmouth[i]>56.237395309385505', 
#             'eyesbottomrightmouth[i]>90.07262877723485', 'eyeopening0[i]>3050.0', 'eyeopening1[i]>4800.0', 
#             'eyeopening3[i]>5550.0', 'eyeschinlow[i]>71.94085792944657', 'moutheyes5[i]>4300.0', 'eyeopeninga[i]>3650.0', 
#             'listrighta[i]>253.02123203678315', 'listrighta1[i]>474.3394203269655', 'listlefta2[i]>159.23150719541383',
#             'eyeopeningb[i]>34.5', 'listline[i]>651.0', 'listlefte1[i]>520.7285997031173', 'listrighte2[i]>385.16561844863736',
#             'listrighta3[i]>617.5820105820106', 'listlefta4[i]>121.57866197183097', 'listrighta4[i]>-66.72222222222221', 
#             'listlefte4[i]>507.0748299319728', 'listrighte4[i]>854.5454545454545', 'listeyedif[i]>81.5', 'listmouthbrow[i]>530.5',
#             'totaleyebrowdif[i]>34.0', 'lenratio[i]>385.8013805135207', 'newmet[i]>18.5', 'newmet2[i]>25.5', 'mouriglist[i]>712.5',
#             'mouleflist[i]>729.5', 'eyeopening2[i]>474.63376548372935', 'broweyedif2[i]>13900.0', 
#             'eyesbottomrightmouth[i]<12.300626508462415', 'topnosesidesofmouth[i]<0.6813332167468502', 'eyeopening3a[i]<2300.0',
#             'eyeopening3b[i]<3650.0', 'listlefta[i]<37.97517848090098', 'listrighta[i]<-111.21741496013718', 
#             'listlefta1[i]<81.29224861180141', 'listrighta1[i]<112.88946254454072', 'listlefta2[i]<-113.31669586248768',
#             'listlefte[i]<193.54802267652977', 'listrighte[i]<173.28056876843607', 'listlefte1[i]<232.25', 
#             'listrighte1[i]<184.7901064249113', 'listlefte2[i]<-132.8350559862188', 'listrighte2[i]<-574.4897435897435', 
#             'listlefta3[i]<107.46923076923076', 'listrighta3[i]<233.44949968730458', 'listlefta4[i]<-208.38888888888889',
#             'listrighte3[i]<-47.44230769230771', 'listlefte4[i]<187.07738095238096', 'listbroweye2[i]<35.0', 'newmet[i]<-19.5',
#             'newmet2[i]<-30.0']

#List of manually changed metrics + extra resulted from the procedure above
# metrnames=['eyeopening[i]>1900', 'broweyedif[i]>1800', 'eyeopening0[i]>1700', 'eyeopening1[i]>2400', 'eyeopening3[i]>3000',
#  'moutheyes5[i]>2300', 'eyeopening3b[i]<6000', 'listlefte4[i]>490', 'eyeopeninga[i]>3000', 'listrighta[i]<-100',
#  'listrighte3[i]<40', 'listrighte[i]<200', 'totaleyebrowdif[i]>32', 'broweyedif2[i]>10000', 'eyeopening3a[i]<3500',
#  'listlefta[i]<40', 'listrighta1[i]<110', 'listlefta2[i]>150', 'listlefta2[i]<-80', 'listrighte1[i]<200',
#  'listlefte2[i]<-130', 'listlefte2[i]>370', 'listrighte2[i]<-500', 'listrighte2[i]>400', 'listrighta3[i]>600',
#  'listrighta3[i]<200', 'listlefta4[i]>100', 'listlefta4[i]<-200', 'listrighta4[i]>-100', 'listrighte4[i]>800',
#  'listrighte4[i]<200', 'eyeopeningb[i]>30', 'moutheyes[i]>30', 'eyesnose[i]>50', 'eyestopnose[i]>120',
#  'eyestopleftmouth[i]>60', 'eyesbottomrightmouth[i]>90', 'eyeschinlow[i]>50', 'eyeopening2[i]>500',
#  'newmet[i]>20', 'newmet[i]<-20', 'newmet2[i]>30', 'newmet2[i]<-20', 'mouriglist[i]>700', 'mouleflist[i]>700',
#  'listline[i]>651.0','listlefte1[i]>520.7285997031173','listlefte1[i]<232.25','listeyedif[i]>81.5',
#  'listmouthbrow[i]>530.5','lenratio[i]>385.8013805135207','topnosesidesofmouth[i]<0.6813332167468502',
#  'listlefta1[i]<81.29224861180141','listlefte[i]<193.54802267652977','listlefta3[i]<107.46923076923076',
#  'listlefte4[i]<187.07738095238096', 'listbroweye2[i]<35.0']

#List of manually selected metrics for distinguishing healthy from patients
metrnames=['eyeopening[i]>1900','broweyedif[i]>1800', 'eyeopening0[i]>1700','eyeopening1[i]>2400','eyeopening3[i]>3000',
           'moutheyes5[i]>2300', 'eyeopening3b[i]<6000','listlefte4[i]>490', 'eyeopeninga[i]>3000', 'listrighta[i]<-100',
           'listrighte3[i]<40','listrighte[i]<200','totaleyebrowdif[i]>32', 'broweyedif2[i]>10000','eyeopening3a[i]<3500',
           'listlefta[i]<40','listrighta1[i]<110','listlefta2[i]>150','listlefta2[i]<-80','listrighte1[i]<200',
           'listlefte2[i]<-130','listlefte2[i]>370','listrighte2[i]<-500','listrighte2[i]>400','listrighta3[i]>600',
           'listrighta3[i]<200','listlefta4[i]>100','listlefta4[i]<-200','listrighta4[i]>-100','listrighte4[i]>800',
           'listrighte4[i]<200','eyeopeningb[i]>30','moutheyes[i]>30','eyesnose[i]>50','eyestopnose[i]>120',
           'eyestopleftmouth[i]>60','eyesbottomrightmouth[i]>90','eyeschinlow[i]>50','eyeopening2[i]>500','newmet[i]>20',
           'newmet[i]<-20','newmet2[i]>30','newmet2[i]<-20','mouriglist[i]>700','mouleflist[i]>700','lenratio[i]>300']


#List of manually selected metrics for distinguishing peripheral from central palsy
pointstoplotpat=[]

#Original metrics resulted from the procedure of Algorithm 2
# alg2=['eyeopening[i]>10000.0', 'broweyedif[i]>46550.0', 'moutheyes[i]>535.1370160790185', 
#               'eyesnose[i]>316.70134188202655', 'eyestopnose[i]>193.17473897763853', 'eyestopleftmouth[i]>411.8902222520694',
#               'eyesbottomrightmouth[i]>132.20043591087926', 'topnosesidesofmouth[i]>204.1792087015364', 'eyeopening0[i]>51700.0', 
#               'eyeopening1[i]>30600.0', 'eyeopening3[i]>41400.0', 'broweyedif1[i]>29300.0', 'moutheyes0[i]>11783.333333333336',
#               'moutheyes5[i]>15600.0', 'eyeopening3a[i]>81800.0', 'eyeopening3b[i]>28500.0', 'eyeopeninga[i]>24850.0', 
#               'listlefta[i]>1635.3801266908288', 'listrighta[i]>285.0308018965768', 'listlefta1[i]>4839.0532097097575', 
#               'listrighta1[i]>3311.6966050238702', 'listrighta2[i]>13.706189577120142', 'eyeopeningb[i]>132.0', 
#               'listlefte[i]>2460.948529640835', 'listrighte[i]>1182.5848392795338', 'listline[i]>3102.0',
#               'listlefte1[i]>3295.2878771752435', 'listrighte1[i]>1636.745965295474', 'listrighte2[i]>531.7222222222222',
#               'listlefta3[i]>7976.451546391752', 'listrighta3[i]>6178.718685300207', 'listlefta4[i]>150.0192105263158', 
#               'listrighta4[i]>23.01971326164876', 'listlefte3[i]>418.49881516587675', 'listrighte3[i]>270.65113636363634', 
#               'listlefte4[i]>3304.7484510532836', 'listrighte4[i]>3631.054585152838', 'listeyedif[i]>292.5', 
#               'listbrowdif[i]>1189.5', 'listbroweye[i]>182.5', 'listslrightleft[i]>2112.4999895366564', 'listmouthbrow[i]>2170.0',
#               'totaleyebrowdif[i]>541.0', 'listbroweye2[i]>176.0', 'lenratio[i]>914.7238640272694', 'newmet[i]>37.0', 
#               'newmet2[i]>512.0', 'mouriglist[i]>3018.0', 'mouleflist[i]>3358.5', 'eyeopening2[i]>1766.9073639147664',
#               'broweyedif2[i]>135900.0', 'moutheyes[i]<5.8169177392968265', 'eyesnose[i]<0.10509390075095837',
#               'eyestopnose[i]<1.2749423602673176', 'eyestopleftmouth[i]<1.6357842993486713', 
#               'eyesbottomrightmouth[i]<2.521159702633838', 'topnosesidesofmouth[i]<0.21314073842455628', 
#               'eyeopening1[i]<300.0', 'eyeopening3[i]<250.0', 'eyeschinlow[i]<1.1546685743706462', 
#               'broweyedif1[i]<3070.0', 'moutheyes5[i]<50.0', 'eyeopening3b[i]<350.0', 'listb[i]<1.0547645262970609', 
#               'listrighta[i]<-932.7974172254507', 'listrighta1[i]<84.19406156585353', 'listlefta2[i]<-930.8717274281486', 
#               'listrighta2[i]<-5794.49135923804', 'listlefte[i]<173.14353547592646', 'listlefte1[i]<214.32142857142856', 
#               'listlefte2[i]<-1089.7200587620137', 'listrighte2[i]<-773.0666666666666', 'listlefta3[i]<88.69090909090909', 
#               'listrighta3[i]<176.36247086247087', 'listlefta4[i]<-2393.9444444444443', 'listrighta4[i]<-24911.5', 
#               'listlefte3[i]<36.39962121212122', 'listrighte3[i]<-421.3074239713775', 'listlefte4[i]<147.94535519125685',
#               'listrighte4[i]<103.07843137254902', 'listbrowdif[i]<50.5', 'listslrightleft[i]<10.621665087118146', 
#               'totaleyebrowdif[i]<0.5', 'lenratio[i]<13.30967169476486', 'newmet[i]<-100.0', 'newmet2[i]<-50.0', 
#               'mouleflist[i]<529.0', 'eyeopening2[i]<106.7672627389279', 'broweyedif2[i]<1900.0']

#List of manually changed metrics + extra resulted from the procedure
# metrnamespat=['eyeopening[i]>2700', 'broweyedif[i]>4500', 'eyeopening0[i]>5000',
#  'eyeopening1[i]>8000', 'broweyedif1[i]>6500', 'broweyedif1[i]<3300', 'eyeopening3[i]<380', 'moutheyes0[i]>2000',
#  'eyeopeninga[i]>5500', 'moutheyes[i]>120', 'eyesnose[i]>60', 'eyestopleftmouth[i]>80', 'topnosesidesofmouth[i]>110',
#  'eyesbottomrightmouth[i]>105', 'eyeopening2[i]>630', 'listline[i]>700', 'listlefte4[i]>570', 'listbroweye[i]>90',
#  'eyeopening3a[i]>15500', 'listrighta2[i]>-10', 'listrighte2[i]>500', 'listrighte3[i]>250', 'broweyedif2[i]<2000',
#  'listrighta4[i]<-1600', 'listbrowdif[i]>140', 'lenratio[i]<20', 'eyeopening3[i]>9000', 'moutheyes5[i]>9500',
#  'listrighta[i]<-150', 'newmet[i]>25', 'newmet[i]<-25', 'newmet2[i]>50', 'lenratio[i]>550', 'broweyedif2[i]>22000',
#  'eyeopeningb[i]>52', 'eyeopening3b[i]>17000', 'eyeopening3b[i]<400', 'listslrightleft[i]>550', 'listmouthbrow[i]>580',
#  'totaleyebrowdif[i]>50', 'listbroweye2[i]>150', 'listrighta1[i]<90', 'listlefte2[i]<-160', 'listrighte2[i]<-700',
#  'listrighta3[i]>730', 'listrighta3[i]<200', 'listlefta4[i]<-300', 'listlefte3[i]>330', 'listlefte3[i]<60',
#  'listrighte3[i]<-200', 'listrighte4[i]<200', 'listlefta1[i]>350', 'listlefte1[i]<220', 'listeyedif[i]>90',
#  'listbrowdif[i]<50', 'eyeopeningb[i]>50', 'eyestopnose[i]>135', 'eyeopening2[i]<100',
#  'listlefta[i]>1635.3801266908288', 'listrighta[i]>285.0308018965768','listrighta1[i]>3311.6966050238702',
#  'listlefte[i]>2460.948529640835', 'listrighte[i]>1182.5848392795338','listlefte1[i]>3295.2878771752435', 
#  'listrighte1[i]>1636.745965295474','listlefta3[i]>7976.451546391752','listlefta4[i]>150.0192105263158',
#  'listrighta4[i]>23.01971326164876','listrighte4[i]>3631.054585152838', 
#  'listeyedif[i]>292.5','mouriglist[i]>3018.0', 'mouleflist[i]>3358.5','moutheyes[i]<5.8169177392968265', 
#  'eyesnose[i]<0.10509390075095837', 'eyestopnose[i]<1.2749423602673176', 'eyestopleftmouth[i]<1.6357842993486713',
#  'eyesbottomrightmouth[i]<2.521159702633838', 'topnosesidesofmouth[i]<0.21314073842455628', 'eyeopening1[i]<300.0',
#  'eyeschinlow[i]<1.1546685743706462','moutheyes5[i]<50.0','listb[i]<1.0547645262970609','listlefta2[i]<-930.8717274281486', 
#  'listrighta2[i]<-5794.49135923804', 'listlefte[i]<173.14353547592646','listlefta3[i]<88.69090909090909',
#  'listslrightleft[i]<10.621665087118146', 'totaleyebrowdif[i]<0.5','newmet2[i]<-50.0', 'mouleflist[i]<529.0']

#List of manually selected metrics for distinguishing central from peripheral palsy
metrnamespat=['eyeopening[i]>2700','broweyedif[i]>4500','eyeopening0[i]>5000','eyeopening1[i]>8000','eyeopening3[i]>9000',
              'eyeopening3[i]<380','broweyedif1[i]>6500','broweyedif1[i]<3300','moutheyes0[i]>2000','eyeopeninga[i]>5500',
              'listrighta[i]<-150','moutheyes[i]>120','eyesnose[i]>60','topnosesidesofmouth[i]>110','eyestopleftmouth[i]>80',
              'eyesbottomrightmouth[i]>105','eyeopening2[i]>630','listline[i]>700','listlefte4[i]>570','listlefte4[i]<150',
              'listbroweye[i]>90','newmet[i]>25','newmet[i]<-25','newmet2[i]>50','lenratio[i]>550','lenratio[i]<20',
              'moutheyes5[i]>9500','broweyedif2[i]>22000','broweyedif2[i]<2000','eyeopening3b[i]>17000','eyeopening3b[i]<400',
              'eyeopening3a[i]>15500','listslrightleft[i]>550','listmouthbrow[i]>580','totaleyebrowdif[i]>50',
              'listbroweye2[i]>150','listrighta1[i]<90','listrighta2[i]>-10','listlefte2[i]<-160','listrighte2[i]>500',
              'listrighte2[i]<-700','listrighta3[i]>730','listrighta3[i]<200','listlefta4[i]<-300','listrighta4[i]<-1600',
              'listlefte3[i]>330','listlefte3[i]<60','listrighte3[i]>250','listrighte3[i]<-200','listrighte4[i]<200',
              'listlefta1[i]>350','listlefte1[i]<220','listeyedif[i]>90','listbrowdif[i]<50','listbrowdif[i]>140',
              'eyeopeningb[i]>50','eyestopnose[i]>135','eyeopening2[i]<100']


#Below are for automatic annotation - Only the manually selected are presented

#Distinguish healthy from patients
metrnames2=['moutheyes[i]>30','eyestopnose[i]>110','eyeschinlow[i]>35','mouriglist[i]>700','broweyedif[i]>1800',
            'eyeopeninga[i]>3000','listline[i]>650','listbroweye2[i]<35','broweyedif1[i]<3200','listlefte4[i]>470',
            'listlefte4[i]<200','listrighta1[i]>460','listrighta1[i]<110','listlefta2[i]>150','eyeopening1[i]>2400',
            'eyeopening3[i]>2900','moutheyes5[i]>2200','eyeopening3b[i]>19000','eyeopening3b[i]<6300','listline[i]<450',
            'listmouthbrow[i]>520','listlefte[i]<200','listbrowdif[i]>140','eyesbottomrightmouth[i]>85',
            'eyesbottomrightmouth[i]<15','eyeopening[i]>1600','eyeopening0[i]>2000','broweyedif2[i]>7500',
            'listrighta[i]<-100','eyeopening3a[i]<3500','totaleyebrowdif[i]>34','listbroweye2[i]<35','listlefta[i]<45',
            'listrighte2[i]<-480','listlefta3[i]<100','listlefta4[i]>110','listrighta4[i]<-1700','listrighte4[i]>800',
            'eyesnose[i]>42','eyestopleftmouth[i]>55','eyeopening2[i]>450','newmet2[i]>20','newmet2[i]<-20','mouleflist[i]>700']

#Distinguishing peripheral from central palsy
metrnamespat2=['eyeopening[i]>1400','eyeopening1[i]>4100','eyeopening3[i]>5900','broweyedif1[i]>6600','broweyedif1[i]<3300',
               'moutheyes5[i]>4200','eyeopeninga[i]>2600','listline[i]>650','eyeopening3a[i]>15500','listlefte2[i]<-115',
               'broweyedif[i]>3200','broweyedif2[i]>15500','listlefte4[i]>460','listbroweye2[i]>120','listlefta1[i]<100',
               'listrighte3[i]>270','listrighte3[i]<20','eyeopeningb[i]>23','eyeopening0[i]>2600','listslrightleft[i]>490',
               'listslrightleft[i]<13','listmouthbrow[i]>550','moutheyes0[i]>1980','listlefta2[i]<-47','eyeopening3b[i]>18000',
               'listlefta[i]<50','listlefte[i]<180','listlefta3[i]<120','listlefte3[i]<80','moutheyes[i]>70',
               'eyestopleftmouth[i]>55','eyesbottomrightmouth[i]>100','eyeopening2[i]>420','newmet[i]>10','newmet2[i]<-30',
               'newmet2[i]>20','lenratio[i]>150']


#These lists and numbers are useful when plotting points at the end
senspoints=[]
senspointspat=[]
bestmetrics=[]
totaltrues=0
initac=0

#Conditions for the loop below
check_pat=args["check_patient"]
if int(manual_an)==1 and int(check_pat)==1: #If check the patients
    pat=0
    pat1=len(metrnamespat)
    heal=len(metrnames)-1
    heal1=len(metrnames)
elif int(manual_an)==1 and int(check_pat)==0: #If check healthy
    pat=len(metrnamespat)-1
    pat1=len(metrnamespat)
    heal=0
    heal1=len(metrnames)
elif int(manual_an)==0 and int(check_pat)==1: #If check the patients
    pat=0
    pat1=len(metrnamespat2)
    heal=len(metrnames2)-1
    heal1=len(metrnames2)
elif int(manual_an)==0 and int(check_pat)==0: #If check healthy
    pat=len(metrnamespat2)-1
    pat1=len(metrnamespat2)
    heal=0
    heal1=len(metrnames2)    
    
    
#Create confusion matrix based on the kind of paralysis
   
    #The two loops check for auto annotation
# for p in range(len(inequalpat2)-1,len(inequalpat2)):
#     for l in range(len(inequal2)):#range(len(inequal)):
#The two loops check for manual annotation
for p in range(pat,pat1):
    for l in range(heal,heal1):
# for p in range(len(metrnamespat)):
#     for l in range(len(metrnames)-1,len(metrnames)):#range(len(inequal)-1,len(inequal)):
        for path in pathnew: #Loop over the paths that contain images (peripheral, central and healthy)
            #The list below only need to initialized for algorithm 2 threshold check
            # metricslist=['eyeopening[i]','broweyedif[i]','moutheyes[i]','eyesnose[i]','eyestopnose[i]','eyestopleftmouth[i]','eyesbottomrightmouth[i]','topnosesidesofmouth[i]',
            #     'eyeopening0[i]','eyeopening1[i]','eyeopening3[i]','eyeschinlow[i]','broweyedif1[i]','moutheyes0[i]','moutheyes5[i]','eyeopening3a[i]','eyeopening3b[i]',
            #     'eyeopeninga[i]','listlefta[i]','listb[i]','listrighta[i]','listlefta1[i]','listrighta1[i]','listlefta2[i]','listrighta2[i]','eyeopeningb[i]','listlefte[i]','listrighte[i]',
            #     'listline[i]','listlefte1[i]','listrighte1[i]','listlefte2[i]','listrighte2[i]','listlefta3[i]','listrighta3[i]','listlefta4[i]','listrighta4[i]','listlefte3[i]','listrighte3[i]',
            #     'listlefte4[i]','listrighte4[i]','listeyedif[i]','listbrowdif[i]','listbroweye[i]','listslrightleft[i]','listmouthbrow[i]','totaleyebrowdif[i]','listbroweye2[i]',
            #     'lenratio[i]','newmet[i]','newmet2[i]','mouriglist[i]','mouleflist[i]','eyeopening2[i]','broweyedif2[i]']
            
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
            lenratio=[]
            
            #The following lists are only used for distinguishing palsy type
            newmet=[]
            newmet2=[]
            mouriglist=[]
            mouleflist=[]
            
            os.chdir(path) #Go into that path
        
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
        
        #The following should only be activated for manually annotated landmarks! Otherwise deactivate!        
        #Only for manual annotated landmarks reshaping should take place since they were created for 900*900 image
                    if int(manual_an)==1:
                        if path==pathper or path==pathcen: #For peripheral or central images do the following:
                            resiz=cv2.resize(image,(900,900)) #Resize image to the same size as when manually annotated
                            fin=image.shape #Get the dimensions of the final image that we want
                            init=resiz.shape #Get the dimensions of the initial image before any transformation
                            lands[:,0]=lands[:,0]*fin[1]/init[1] #Transform the x coordinates of the landmarks
                            lands[:,1]=lands[:,1]*fin[0]/init[0] #Transform the y coordinates of the landmarks
                            lands=lands.astype(int) #Make the landmarks integers since transformation results in float numbers
                #The following are just to confirm that landmarks are transformed properly
                        # for (x,y) in lands:
                        #     cv2.circle(image, (x,y),1,(0,0,255),-1)
                        # cv2.imshow("Cropped Image", image)
                        # cv2.waitKey(0)
        #From now on it is for both manually and automatically annotated image
                            
                 #As a first step the face is cropped and the landmarks are adapted to fit in the size of that crop   
                    for i in range(len(facebox)): #Loop over the facebox coordinates
                        if facebox[i]<0: #If a coordinate is outside of the image set it to 0
                            facebox[i]=0
                    cropped_img = image[facebox[1]:facebox[3],facebox[0]:facebox[2]] #Crop image in x and y direction to include the face only
                    
                    #The following are just to confirm that the image is cropped properly (It will be)
                    # cv2.imshow("Cropped Image", cropped_img)
                    # cv2.waitKey(0)
                            
                    lands[:,0]=lands[:,0]-facebox[0] #Transform the x coordinates of the landmarks based on the cropped image
                    lands[:,1]=lands[:,1]-facebox[1] #Transform the y coordinates of the landmarks based on the cropped image
        
                    #The following are just to confirm that the landmarks are transformed properly
                    # for (x,y) in lands:
                    #     cv2.circle(cropped_img, (x,y),1,(0,0,255),-1)
                    # cv2.imshow("Cropped Image", cropped_img)
                    # cv2.waitKey(0)
         
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
                      cv2.circle(resi, (x,y),3,(0,0,255),-1)
                    cv2.imwrite(filename[:-8]+'norot'+lastones, resi)
                    # cv2.imshow("Cropped Image", cropped_img)
                    # cv2.waitKey(0)
            
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
                    
                    
           # Measure 19: Distance between lowest point on chin and almost top of nose         
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
                    listb.append(100*np.abs(b1-b)) #Add that difference for the specific image to a list
                    
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
                
                
           # Measures 31 and 32: Best line fit of 3 most left points of each eye and keep the slope of each brow to a list
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
                    #mouthhigh=np.max(listmouthhighlow)
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
               
            
        #for j in range(40): #Use that if we want to check for the best threshold. Better not to do that since there are many
        # possible combination of values and will take extremely long

            # #Create confusion matrix based on the kind of paralysis
            # for l in range(len(inequal)):
            if path==pathper or path==pathcen: #If we are in the folders with peripheral or central palsy pictures then:           
                #Below normal/patients are distinguished
                tp=0 #Initialize True Positives to 0
                fn=0 #Initialize False Negatives to 0
                for i in range(len(eyeopening2)):# Loop over the number of images in a folder
                    #Conditions that if hold true then an image is classified correctly as 'patient'
                    #Only automatic annotation of healthy individuals is performed
                    if int(manual_an)==1: #These are the threshods for manual annotation
                        if any([eval(metrnames[k]) for k in range(l+1)])==True: 
                        #Above are the best thresholds found 
                            tp=tp+1
                        else: #If all the above conditions are false then the image is incorrectly classified as 'normal'
                            fn=fn+1
                    else: #These are the threshods for automatic annotation
                        if any([eval(metrnames2[k]) for k in range(l+1)])==True:
                            tp=tp+1
                        else: #If all the above conditions are false then the image is incorrectly classified as 'normal'
                            fn=fn+1
                        
                #Append the results to lists
                tplis.append(tp)
                fnlis.append(fn)
                
                if len(tplis)%2==0: #Because in tp and fn both central and peripheral palsy are counted. 
                    #Each time we loop only in one of these folders and so, only one value is appended at a time
                    tplast.append(tplis[-1]+tplis[-2])
                    fnlast.append(fnlis[-1]+fnlis[-2])
                
                #Below we distinguish peripheral from central palsy
                if path==pathper: #If we are in peripheral folder
                    tp1=0 #Initialize True Positives to 0
                    fn1=0 #Initialize False Negatives to 0
                    for i in range(len(eyeopening)): #Loop over the total number of images in that folder
                        if int(manual_an)==1: #For manual annotation the metrics below are used
                            if any([eval(metrnamespat[u]) for u in range(p+1)])==True: 
                                 tp1=tp1+1
                            else:
                                 fn1=fn1+1
                        else: #For automatic annotation the metrics below are used. If auto annotation is good metrics will be the same.
                            if any([eval(metrnamespat2[u]) for u in range(p+1)])==True:
                                tp1=tp1+1
                            else:
                                fn1=fn1+1
                                
                    #Append the results to lists
                    tplis1.append(tp1) 
                    fnlis1.append(fn1)
                    
                elif path==pathcen: #If we are in central folder
                    fp1=0 #Initialize False Positives to 0
                    tn1=0 #Initialze True Negatives to 0
                    for i in range(len(eyeopening)): #Loop over the total number of images in that folder
                        if int(manual_an)==1: #For manual annotation the metrics below are used
                            if any([eval(metrnamespat[u]) for u in range(p+1)])==True: 
                                fp1=fp1+1
                            else:
                                tn1=tn1+1
                        else: #For automatic annotation the metrics below are used. If auto annotation is good metrics will be the same.
                            if any([eval(metrnamespat2[u]) for u in range(p+1)])==True:
                                fp1=fp1+1
                            else:
                                tn1=tn1+1
                                
                    #Append the results to lists
                    fplis1.append(fp1)
                    tnlis1.append(tn1)
        
        #The metrics below are used to distinguish health from patient                
            elif path==pathheal: #If we are in the folder with normal pictures then:
                fp=0 #Initialize False Positives to 0
                tn=0 #Initialze True Negatives to 0
                for i in range(len(eyeopening2)): #Loop over the number of images in that folder
                   #Conditions that if hold true then an image is classified incorrectly as 'patient'
                    #These are the exact same conditions as above!
                    if int(manual_an)==1:
                        if any([eval(metrnames[k]) for k in range(l+1)])==True: 
                            fp=fp+1
                        else: #If all the above conditions are false then the image is correctly classified as 'normal'
                            tn=tn+1
                    else:
                        if any([eval(metrnames2[k]) for k in range(l+1)])==True:
                            fp=fp+1
                        else: #If all the above conditions are false then the image is correctly classified as 'normal'
                            tn=tn+1
                        
                #Append the results to lists
                fplis.append(fp)
                tnlis.append(tn)
            
        
            ################################################
            #Below the metrics are printed. This can also be deactivated. Only useful during calculation of these metrics
            
            if j==1: #Needed in order to print a few of the last metrics. It remained after grid search for thresholds
                
                #Initialize empty lists to be used below
                tnlastfin2=[]
                fplastfin2=[]
        
                if path==pathper or path==pathcen: #Metrics for peripheral and central paths
                    if path==pathper: #Metrics for peripheral folder
                        
                        #The following only need to be activated if we want to use the exact form of algorithm2
                        # metrics=[eyeopening,broweyedif,moutheyes,eyesnose,eyestopnose,eyestopleftmouth,eyesbottomrightmouth,topnosesidesofmouth,
                        #       eyeopening0,eyeopening1,eyeopening3,eyeschinlow,broweyedif1,moutheyes0,moutheyes5,eyeopening3a,eyeopening3b,
                        #       eyeopeninga,listlefta,listb,listrighta,listlefta1,listrighta1,listlefta2,listrighta2,eyeopeningb,listlefte,listrighte,
                        #       listline,listlefte1,listrighte1,listlefte2,listrighte2,listlefta3,listrighta3,listlefta4,listrighta4,listlefte3,listrighte3,
                        #       listlefte4,listrighte4,listeyedif,listbrowdif,listbroweye,listslrightleft,listmouthbrow,totaleyebrowdif,listbroweye2,
                        #       lenratio,newmet,newmet2,mouriglist,mouleflist,eyeopening2,broweyedif2]
                        # maxofmetricsper=[]
                        # minofmetricsper=[]
                        # for index,el in enumerate(metrics):
                        #         maxofmetricsper.append(np.max(el))
                        #         minofmetricsper.append(np.min(el))

                                
                  #The following only need to be activated if we want to manually examine the maximum value of each metric
                  #and to create the tables with their values
                        
                        # print("Below are metrics for peripheral palsy")
                        # print("\n")
                        
                        # print("eyeopening - Measure 8: Calculate difference in amount of eyelid opening between eyes (1)")
                        # print("Maximum for peripheral")
                        # print(np.max(eyeopening))
                        # print("Minimum for peripheral")
                        # print(np.min(eyeopening))
                        # print("Mean for peripheral")
                        # print(np.mean(eyeopening))
                        # print("Std for peripheral")
                        # print(np.std(eyeopening))
                        
                        # print("broweyedif - Measure 11: Difference between the difference of the") 
                        # print("distance of the lowest point of each eye to the highest of the corresponding brow")
                        # print("Maximum for peripheral")
                        # print(np.max(broweyedif))
                        # print("Minimum for peripheral")
                        # print(np.min(broweyedif))
                        # print("Mean for peripheral")
                        # print(np.mean(broweyedif))
                        # print("Std for peripheral")
                        # print(np.std(broweyedif))
                        
                        # print("eyeopening0 - Measure 12: Difference of distance between tip of nose and each brow")
                        # print("Maximum for peripheral")
                        # print(np.max(eyeopening0))
                        # print("Minimum for peripheral")
                        # print(np.min(eyeopening0))
                        # print("Mean for peripheral")
                        # print(np.mean(eyeopening0))
                        # print("Std for peripheral")
                        # print(np.std(eyeopening0))
                        
                        # print("eyeopening1 - Measure 13: Difference between most-left point of nose and each brow")
                        # print("Maximum for peripheral")
                        # print(np.max(eyeopening1))
                        # print("Minimum for peripheral")
                        # print(np.min(eyeopening1))
                        # print("Mean for peripheral")
                        # print(np.mean(eyeopening1))
                        # print("Std for peripheral")
                        # print(np.std(eyeopening1))
                        
                        # print("eyeopening3 - Measure 14: Distance between sides of mouth and corresponding brows")
                        # print("Maximum for peripheral")
                        # print(np.max(eyeopening3))
                        # print("Minimum for peripheral")
                        # print(np.min(eyeopening3))
                        # print("Mean for peripheral")
                        # print(np.mean(eyeopening3))
                        # print("Std for peripheral")
                        # print(np.std(eyeopening3))
                        
                        # # print("eyeopening6 - Measure 18: Difference of eyebrow heights")
                        # # print("Maximum for peripheral")
                        # # print(np.max(eyeopening6))
                        # # print("Minimum for peripheral")
                        # # print(np.min(eyeopening6))
                        # # print("Mean for peripheral")
                        # # print(np.mean(eyeopening6))
                        # # print("Std for peripheral")
                        # # print(np.std(eyeopening6))
                        
                        # print("broweyedif1 - Measure 19: Distance between lowest point on chin and almost top of nose")
                        # print("Maximum for peripheral")
                        # print(np.max(broweyedif1))
                        # print("Minimum for peripheral")
                        # print(np.min(broweyedif1))
                        # print("Mean for peripheral")
                        # print(np.mean(broweyedif1))
                        # print("Std for peripheral")
                        # print(np.std(broweyedif1))  
                        
                        # print("moutheyes5 - Measure 22: Calculate difference of opening of each side of the mouth")
                        # print("Maximum for peripheral")
                        # print(np.max(moutheyes5))
                        # print("Minimum for peripheral")
                        # print(np.min(moutheyes5))
                        # print("Mean for peripheral")
                        # print(np.mean(moutheyes5))
                        # print("Std for peripheral")
                        # print(np.std(moutheyes5))               
                        
                        # print("broweyedif2 - Sum all distances from the brow to mouth, nose, chin etc. - Added metric in loop")
                        # print("Maximum for peripheral")
                        # print(np.max(broweyedif2))
                        # print("Minimum for peripheral")
                        # print(np.min(broweyedif2))
                        # print("Mean for peripheral")
                        # print(np.mean(broweyedif2))
                        # print("Std for peripheral")
                        # print(np.std(broweyedif2))
                        
                        # # print("listbrowslopedif - Measure 17: Distance between lowest point on chin and brows")
                        # # print("Maximum for peripheral")
                        # # print(np.max(listbrowslopedif))
                        # # print("Minimum for peripheral")
                        # # print(np.min(listbrowslopedif))
                        # # print("Mean for peripheral")
                        # # print(np.mean(listbrowslopedif))
                        # # print("Std for peripheral")
                        # # print(np.std(listbrowslopedif))               
                        
                        # print("moutheyes0 - Measures 20: Differences of the average of 3 left from the average of 3 right points")
                        # print("on the top of each eye")
                        # print("Maximum for peripheral")
                        # print(np.max(moutheyes0))
                        # print("Minimum for peripheral")
                        # print(np.min(moutheyes0))
                        # print("Mean for peripheral")
                        # print(np.mean(moutheyes0))
                        # print("Std for peripheral")
                        # print(np.std(moutheyes0))
                        
                        
                        # print("eyeopening3b - Measure 16: Distance between sides of mouth and corresponding brows (try to amplify it)")
                        # print("Maximum for peripheral")
                        # print(np.max(eyeopening3b))
                        # print("Minimum for peripheral")
                        # print(np.min(eyeopening3b))
                        # print("Mean for peripheral")
                        # print(np.mean(eyeopening3b))
                        # print("Std for peripheral")
                        # print(np.std(eyeopening3b))   
                        
                        # print("eyeopeninga - Measures 9 and 10: A second and third way to calculate amount of eyelid opening")
                        # print("Maximum for peripheral")
                        # print(np.max(eyeopeninga))
                        # print("Minimum for peripheral")
                        # print(np.min(eyeopeninga))
                        # print("Mean for peripheral")
                        # print(np.mean(eyeopeninga))
                        # print("Std for peripheral")
                        # print(np.std(eyeopeninga))  
                        
                        # print("listrighta - Measures 23, 24, 25: Fit the best lines on left and right brows")
                        # print("Maximum for peripheral")
                        # print(np.max(listrighta))
                        # print("Minimum for peripheral")
                        # print(np.min(listrighta))
                        # print("Mean for peripheral")
                        # print(np.mean(listrighta))
                        # print("Std for peripheral")
                        # print(np.std(listrighta))
                        
                        # print("listline - Measure 34: Distance from top of nose to the farthest point of each eye ")
                        # print("Maximum for peripheral")
                        # print(np.max(listline))
                        # print("Minimum for peripheral")
                        # print(np.min(listline))
                        # print("Mean for peripheral")
                        # print(np.mean(listline))
                        # print("Std for peripheral")
                        # print(np.std(listline))
                        
                        # print("listlefte4 - Measures 46 and 47: Best line fit of the 2 most right bottom points of each eye") 
                        # print("and keep slopes")
                        # print("Maximum for peripheral")
                        # print(np.max(listlefte4))
                        # print("Minimum for peripheral")
                        # print(np.min(listlefte4))
                        # print("Mean for peripheral")
                        # print(np.mean(listlefte4))
                        # print("Std for peripheral")
                        # print(np.std(listlefte4))
                        
                        # print("listbroweye - Measure 49, 50, 51: Difference of the highest point of both brows")
                        # print("from the lowest of both brows")
                        # print("Maximum for peripheral")
                        # print(np.max(listbroweye))
                        # print("Minimum for peripheral")
                        # print(np.min(listbroweye))
                        # print("Mean for peripheral")
                        # print(np.mean(listbroweye))
                        # print("Std for peripheral")
                        # print(np.std(listbroweye))
                        
                        # print("eyeopening3a - Measure 15: Distance between sides of mouth and corresponding brows (try to amplify it)")
                        # print("Maximum for peripheral")
                        # print(np.max(eyeopening3a))
                        # print("Minimum for peripheral")
                        # print(np.min(eyeopening3a))
                        # print("Mean for peripheral")
                        # print(np.mean(eyeopening3a))
                        # print("Std for peripheral")
                        # print(np.std(eyeopening3a))
                        
                        
                        # print("listslrightleft - Measures 35, 36 and 37: Best line fit of 2 most left points of each eye")
                        # print("and keep slope and difference of slopes to a list")
                        # print("Maximum for peripheral")
                        # print(np.max(listslrightleft))
                        # print("Minimum for peripheral")
                        # print(np.min(listslrightleft))
                        # print("Mean for peripheral")
                        # print(np.mean(listslrightleft))
                        # print("Std for peripheral")
                        # print(np.std(listslrightleft))
                        
                        
                        # print("listmouthbrow - Measure 52: Difference of the highest from the lowest point of the 4 most-left")
                        # print("and 4 most-right points of the mouth")
                        # print("Maximum for peripheral")
                        # print(np.max(listmouthbrow))
                        # print("Minimum for peripheral")
                        # print(np.min(listmouthbrow))
                        # print("Mean for peripheral")
                        # print(np.mean(listmouthbrow))
                        # print("Std for peripheral")
                        # print(np.std(listmouthbrow))
                        
                        
                        # print("totaleyebrowdif - Measure 26: Calculate difference of top points in each brow and eye from each other")
                        # print("Maximum for peripheral")
                        # print(np.max(totaleyebrowdif))
                        # print("Minimum for peripheral")
                        # print(np.min(totaleyebrowdif))
                        # print("Mean for peripheral")
                        # print(np.mean(totaleyebrowdif))
                        # print("Std for peripheral")
                        # print(np.std(totaleyebrowdif))
                        
                        
                        # print("listbroweye2 - Measure 49, 50, 51: Difference of the highest point of both brows") 
                        # print("from the lowest of both brows")
                        # print("Maximum for peripheral")
                        # print(np.max(listbroweye2))
                        # print("Minimum for peripheral")
                        # print(np.min(listbroweye2))
                        # print("Mean for peripheral")
                        # print(np.mean(listbroweye2))
                        # print("Std for peripheral")
                        # print(np.std(listbroweye2))
                        
                        # print("listlefta - Measures 23, 24, 25: Fit the best lines on left and right brows")
                        # print("Maximum for peripheral")
                        # print(np.max(listlefta))
                        # print("Minimum for peripheral")
                        # print(np.min(listlefta))
                        # print("Mean for peripheral")
                        # print(np.mean(listlefta))
                        # print("Std for peripheral")
                        # print(np.std(listlefta))
                        
                        # print("listlefta1 - Measures 27 and 28: Best line fit of 3 most left points of each brow")
                        # print("and save the slope of each brow to a list")
                        # print("Maximum for peripheral")
                        # print(np.max(listlefta1))
                        # print("Minimum for peripheral")
                        # print(np.min(listlefta1))
                        # print("Mean for peripheral")
                        # print(np.mean(listlefta1))
                        # print("Std for peripheral")
                        # print(np.std(listlefta1))
                        
                        # print("listrighta1 - Measures 27 and 28: Best line fit of 3 most left points of each brow")
                        # print("and save the slope of each brow to a list")
                        # print("Maximum for peripheral")
                        # print(np.max(listrighta1))
                        # print("Minimum for peripheral")
                        # print(np.min(listrighta1))
                        # print("Mean for peripheral")
                        # print(np.mean(listrighta1))
                        # print("Std for peripheral")
                        # print(np.std(listrighta1))
                        
                        # print("listlefta2 - Measures 29 and 30: Best line fit of 3 middle points of each brow")
                        # print("and keep the slope of each one of them to a list")
                        # print("Maximum for peripheral")
                        # print(np.max(listlefta2))
                        # print("Minimum for peripheral")
                        # print(np.min(listlefta2))
                        # print("Mean for peripheral")
                        # print(np.mean(listlefta2))
                        # print("Std for peripheral")
                        # print(np.std(listlefta2))
                        
                        # print("listrighta2 - Measures 29 and 30: Best line fit of 3 middle points of each brow")
                        # print("and keep the slope of each one of them to a list")
                        # print("Maximum for peripheral")
                        # print(np.max(listrighta2))
                        # print("Minimum for peripheral")
                        # print(np.min(listrighta2))
                        # print("Mean for peripheral")
                        # print(np.mean(listrighta2))
                        # print("Std for peripheral")
                        # print(np.std(listrighta2))
                        
                        # print("listb - Measures 23, 24, 25: Fit the best lines on left and right brows")
                        # print("Maximum for peripheral")
                        # print(np.max(listb))
                        # print("Minimum for peripheral")
                        # print(np.min(listb))
                        # print("Mean for peripheral")
                        # print(np.mean(listb))
                        # print("Std for peripheral")
                        # print(np.std(listb))
                        
                        # print("listlefte - Measures 31, 32 and 33: Best line fit of 3 most left points of each eye")
                        # print("and keep the slope of each brow to a list")
                        # print("Maximum for peripheral")
                        # print(np.max(listlefte))
                        # print("Minimum for peripheral")
                        # print(np.min(listlefte))
                        # print("Mean for peripheral")
                        # print(np.mean(listlefte))
                        # print("Std for peripheral")
                        # print(np.std(listlefte))
                        
                        # print("listrighte - Measures 31, 32 and 33: Best line fit of 3 most left points of each eye")
                        # print("and keep the slope of each brow to a list")
                        # print("Maximum for peripheral")
                        # print(np.max(listrighte))
                        # print("Minimum for peripheral")
                        # print(np.min(listrighte))
                        # print("Mean for peripheral")
                        # print(np.mean(listrighte))
                        # print("Std for peripheral")
                        # print(np.std(listrighte))
                        
        
                        # print("listlefte1 - Measures 35, 36 and 37: Best line fit of 2 most left points of each eye")
                        # print("and keep slope and difference of slopes to a list")
                        # print("Maximum for peripheral")
                        # print(np.max(listlefte1))
                        # print("Minimum for peripheral")
                        # print(np.min(listlefte1))
                        # print("Mean for peripheral")
                        # print(np.mean(listlefte1))
                        # print("Std for peripheral")
                        # print(np.std(listlefte1))
                        
                        # print("listrighte1 - Measures 35, 36 and 37: Best line fit of 2 most left points of each eye")
                        # print("and keep slope and difference of slopes to a list")
                        # print("Maximum for peripheral")
                        # print(np.max(listrighte1))
                        # print("Minimum for peripheral")
                        # print(np.min(listrighte1))
                        # print("Mean for peripheral")
                        # print(np.mean(listrighte1))
                        # print("Std for peripheral")
                        # print(np.std(listrighte1))
                        
                        # print("listlefte2 - Measures 38 and 39: Best line fit of the 2 most right points of each eye")
                        # print("and keep slope in a list")
                        # print("Maximum for peripheral")
                        # print(np.max(listlefte2))
                        # print("Minimum for peripheral")
                        # print(np.min(listlefte2))
                        # print("Mean for peripheral")
                        # print(np.mean(listlefte2))
                        # print("Std for peripheral")
                        # print(np.std(listlefte2))
                        
                        # print("listrighte2 - Measures 38 and 39: Best line fit of the 2 most right points of each eye")
                        # print("and keep slope in a list")
                        # print("Maximum for peripheral")
                        # print(np.max(listrighte2))
                        # print("Minimum for peripheral")
                        # print(np.min(listrighte2))
                        # print("Mean for peripheral")
                        # print(np.mean(listrighte2))
                        # print("Std for peripheral")
                        # print(np.std(listrighte2))
                        
                        # print("listlefta3 - Measures 40 and 41: Best line fit of the 2 most left points of each brow")
                        # print("and keep slopes in a list")
                        # print("Maximum for peripheral")
                        # print(np.max(listlefta3))
                        # print("Minimum for peripheral")
                        # print(np.min(listlefta3))
                        # print("Mean for peripheral")
                        # print(np.mean(listlefta3))
                        # print("Std for peripheral")
                        # print(np.std(listlefta3))
                        
                        # print("listrighta3 - Measures 40 and 41: Best line fit of the 2 most left points of each brow")
                        # print("and keep slopes in a list")
                        # print("Maximum for peripheral")
                        # print(np.max(listrighta3))
                        # print("Minimum for peripheral")
                        # print(np.min(listrighta3))
                        # print("Mean for peripheral")
                        # print(np.mean(listrighta3))
                        # print("Std for peripheral")
                        # print(np.std(listrighta3))
                        
                        # print("listlefta4 - Measures 42 and 43: Best line fit of the 2 most right points of each brow")
                        # print("and keep slopes in a list")
                        # print("Maximum for peripheral")
                        # print(np.max(listlefta4))
                        # print("Minimum for peripheral")
                        # print(np.min(listlefta4))
                        # print("Mean for peripheral")
                        # print(np.mean(listlefta4))
                        # print("Std for peripheral")
                        # print(np.std(listlefta4))
                        
                        # print("listrighta4 - Measures 42 and 43: Best line fit of the 2 most right points of each brow")
                        # print("and keep slopes in a list")
                        # print("Maximum for peripheral")
                        # print(np.max(listrighta4))
                        # print("Minimum for peripheral")
                        # print(np.min(listrighta4))
                        # print("Mean for peripheral")
                        # print(np.mean(listrighta4))
                        # print("Std for peripheral")
                        # print(np.std(listrighta4))
                        
                        # print("listlefte3 - Measures 44 and 45: Best line fit of the 2 most left bottom points of each eye")
                        # print("and keep slopes")
                        # print("Maximum for peripheral")
                        # print(np.max(listlefte3))
                        # print("Minimum for peripheral")
                        # print(np.min(listlefte3))
                        # print("Mean for peripheral")
                        # print(np.mean(listlefte3))
                        # print("Std for peripheral")
                        # print(np.std(listlefte3))
                        
                        # print("listrighte3 - Measures 44 and 45: Best line fit of the 2 most left bottom points of each eye")
                        # print("and keep slopes")
                        # print("Maximum for peripheral")
                        # print(np.max(listrighte3))
                        # print("Minimum for peripheral")
                        # print(np.min(listrighte3))
                        # print("Mean for peripheral")
                        # print(np.mean(listrighte3))
                        # print("Std for peripheral")
                        # print(np.std(listrighte3))
                        
                        
                        # print("listrighte4 - Measures 46 and 47: Best line fit of the 2 most right bottom points of each eye")
                        # print("and keep slopes")
                        # print("Maximum for peripheral")
                        # print(np.max(listrighte4))
                        # print("Minimum for peripheral")
                        # print(np.min(listrighte4))
                        # print("Mean for peripheral")
                        # print(np.mean(listrighte4))
                        # print("Std for peripheral")
                        # print(np.std(listrighte4))
                        
                        # print("listeyedif - Measure 48: Difference of the highest point of both eyes from the lowest of both eyes")
                        # print("Maximum for peripheral")
                        # print(np.max(listeyedif))
                        # print("Minimum for peripheral")
                        # print(np.min(listeyedif))
                        # print("Mean for peripheral")
                        # print(np.mean(listeyedif))
                        # print("Std for peripheral")
                        # print(np.std(listeyedif))
                        
                        # print("listbrowdif - Measure 49, 50, 51: Difference of the highest point of both brows")
                        # print("from the lowest of both brows")
                        # print("Maximum for peripheral")
                        # print(np.max(listbrowdif))
                        # print("Minimum for peripheral")
                        # print(np.min(listbrowdif))
                        # print("Mean for peripheral")
                        # print(np.mean(listbrowdif))
                        # print("Std for peripheral")
                        # print(np.std(listbrowdif))
                        
                        # print("eyeopeningb - Measures 9 and 10: A second and third way to calculate amount of eyelid opening")
                        # print("Maximum for peripheral")
                        # print(np.max(eyeopeningb))
                        # print("Minimum for peripheral")
                        # print(np.min(eyeopeningb))
                        # print("Mean for peripheral")
                        # print(np.mean(eyeopeningb))
                        # print("Std for peripheral")
                        # print(np.std(eyeopeningb))
                        
        
                        
                        # print('\n')
                        # print('BELOW ARE LISTS THAT ARE MAINLY USED TO DISTINGUISH NORMAL/PATIENT')
                        
                        # print("moutheyes -  Measure 1: Distance from centre of eyes to edges of mouth")
                        # print("Maximum for peripheral")
                        # print(np.max(moutheyes))
                        # print("Minimum for peripheral")
                        # print(np.min(moutheyes))
                        # print("Mean for peripheral")
                        # print(np.mean(moutheyes))
                        # print("Std for peripheral")
                        # print(np.std(moutheyes))
                        
                        # print("eyesnose - Measure 2: Distance from centre of eyes to sides of the nose")
                        # print("Maximum for peripheral")
                        # print(np.max(eyesnose))
                        # print("Minimum for peripheral")
                        # print(np.min(eyesnose))
                        # print("Mean for peripheral")
                        # print(np.mean(eyesnose))
                        # print("Std for peripheral")
                        # print(np.std(eyesnose))
                        
                        # print("eyestopnose - Measure 3: Euclidean Distance of each eye from the top of the nose")
                        # print("Maximum for peripheral")
                        # print(np.max(eyestopnose))
                        # print("Minimum for peripheral")
                        # print(np.min(eyestopnose))
                        # print("Mean for peripheral")
                        # print(np.mean(eyestopnose))
                        # print("Std for peripheral")
                        # print(np.std(eyestopnose))
                        
                        # print("eyestopleftmouth - Measure 4: Euclidean Distance between each eye and the top left corner of the mouth")
                        # print("Maximum for peripheral")
                        # print(np.max(eyestopleftmouth))
                        # print("Minimum for peripheral")
                        # print(np.min(eyestopleftmouth))
                        # print("Mean for peripheral")
                        # print(np.mean(eyestopleftmouth))
                        # print("Std for peripheral")
                        # print(np.std(eyestopleftmouth))
                        
                        # print("eyesbottomrightmouth - Measure 5: Distance between eyes and bottom corner of the right side of the mouth")
                        # print("Maximum for peripheral")
                        # print(np.max(eyesbottomrightmouth))
                        # print("Minimum for peripheral")
                        # print(np.min(eyesbottomrightmouth))
                        # print("Mean for peripheral")
                        # print(np.mean(eyesbottomrightmouth))
                        # print("Std for peripheral")
                        # print(np.std(eyesbottomrightmouth))
                        
                        # print("topnosesidesofmouth - Measure 6: Difference in distance between top of nose and each side of the mouth")
                        # print("Maximum for peripheral")
                        # print(np.max(topnosesidesofmouth))
                        # print("Minimum for peripheral")
                        # print(np.min(topnosesidesofmouth))
                        # print("Mean for peripheral")
                        # print(np.mean(topnosesidesofmouth))
                        # print("Std for peripheral")
                        # print(np.std(topnosesidesofmouth))
                        
                        # print("eyeschinlow - Measure 7: Euclidean Distance between each eye and lowest point of the chin")
                        # print("Maximum for peripheral")
                        # print(np.max(eyeschinlow))
                        # print("Minimum for peripheral")
                        # print(np.min(eyeschinlow))
                        # print("Mean for peripheral")
                        # print(np.mean(eyeschinlow))
                        # print("Std for peripheral")
                        # print(np.std(eyeschinlow))  
                        
                        # print("eyeopening2 - Sum all distances from the eye to mouth, nose, chin etc. - Added in a loop")
                        # print("Maximum for peripheral")
                        # print(np.max(eyeopening2))
                        # print("Minimum for peripheral")
                        # print(np.min(eyeopening2))
                        # print("Mean for peripheral")
                        # print(np.mean(eyeopening2))
                        # print("Std for peripheral")           
                        # print(np.std(eyeopening2))  
        
                    
                        # print("newmet - Difference of left and right eye openings")
                        # print("Maximum for peripheral")
                        # print(np.max(newmet))
                        # print("Minimum for peripheral")
                        # print(np.min(newmet))
                        # print("Mean for peripheral")
                        # print(np.mean(newmet))
                        # print("Std for peripheral")
                        # print(np.std(newmet))   
                        
                        # print("newmet2 - Difference of left and right eyebrow heights")
                        # print("Maximum for peripheral")
                        # print(np.max(newmet2))
                        # print("Minimum for peripheral")
                        # print(np.min(newmet2))
                        # print("Mean for peripheral")
                        # print(np.mean(newmet2))
                        # print("Std for peripheral")
                        # print(np.std(newmet2))  
                        
                        # print("mouriglist - lowest point of right eye")
                        # print("Maximum for peripheral")
                        # print(np.max(mouriglist))
                        # print("Minimum for peripheral")
                        # print(np.min(mouriglist))
                        # print("Mean for peripheral")
                        # print(np.mean(mouriglist))
                        # print("Std for peripheral")
                        # print(np.std(mouriglist))  
                        
                        # print("mouleflist - lowest point of left eye")
                        # print("Maximum for peripheral")
                        # print(np.max(mouleflist))
                        # print("Minimum for peripheral")
                        # print(np.min(mouleflist))
                        # print("Mean for peripheral")
                        # print(np.mean(mouleflist))
                        # print("Std for peripheral")
                        # print(np.std(mouleflist))  
        
                        # # print("ratioleft")
                        # # print("Maximum for peripheral")
                        # # print(np.max(ratioleft))
                        # # print("Minimum for peripheral")
                        # # print(np.min(ratioleft))
                        # # print("Mean for peripheral")
                        # # print(np.mean(ratioleft))
                        # # print("Std for peripheral")
                        # # print(np.std(ratioleft))  
                        
                        # # print("ratioright")
                        # # print("Maximum for peripheral")
                        # # print(np.max(ratioright))
                        # # print("Minimum for peripheral")
                        # # print(np.min(ratioright))
                        # # print("Mean for peripheral")
                        # # print(np.mean(ratioright))
                        # # print("Std for peripheral")
                        # # print(np.std(ratioright))  
                        
                        # print("lenratio - ratio of the two eye ratios")
                        # print("Maximum for peripheral")
                        # print(np.max(lenratio))
                        # print("Minimum for peripheral")
                        # print(np.min(lenratio))
                        # print("Mean for peripheral")
                        # print(np.mean(lenratio))
                        # print("Std for peripheral")
                        # print(np.std(lenratio))  
                        
                    
                        #Set them to 0 in order to use them again in next loop
                        tpfin=0
                        fnfin=0
                        fnfinlis=[]
                        
            #Below metrics for the task of distinguishing peripheral/central are calculated
                        print('\n')
                        tpfin1=0 #True Positives set to 0
                        fnfin1=0 #False Negatives set to 0
                        fnfinlis1=[] #Initialize empty list to be used below
                        
                        for i in range(len(eyeopening)): #Loop over the number of peripheral images
                            if int(manual_an)==1: #For manual annotation the below metrics are used to distinguish peripheral/central
                                if any([eval(metrnamespat[u]) for u in range(p+1)])==True: 
                                    tpfin1=tpfin1+1
                                else:
                                    fnfin1=fnfin1+1
                                    fnfinlis1.append(listofnames[i]) #Append the missclassified peripheral palsy as central to a list    
                            else:
                                if any([eval(metrnamespat2[u]) for u in range(p+1)])==True:
                                    tpfin1=tpfin1+1
                                else:
                                    fnfin1=fnfin1+1
                                    fnfinlis1.append(listofnames[i]) #Append the missclassified peripheral palsy as central to a list 
        
        
                    if path==pathcen: #Metrics for Central folder
                        
                        #The following only need to be activated if we want to use the exact form of algorithm2
                        # metrics=[eyeopening,broweyedif,moutheyes,eyesnose,eyestopnose,eyestopleftmouth,eyesbottomrightmouth,topnosesidesofmouth,
                        #       eyeopening0,eyeopening1,eyeopening3,eyeschinlow,broweyedif1,moutheyes0,moutheyes5,eyeopening3a,eyeopening3b,
                        #       eyeopeninga,listlefta,listb,listrighta,listlefta1,listrighta1,listlefta2,listrighta2,eyeopeningb,listlefte,listrighte,
                        #       listline,listlefte1,listrighte1,listlefte2,listrighte2,listlefta3,listrighta3,listlefta4,listrighta4,listlefte3,listrighte3,
                        #       listlefte4,listrighte4,listeyedif,listbrowdif,listbroweye,listslrightleft,listmouthbrow,totaleyebrowdif,listbroweye2,
                        #       lenratio,newmet,newmet2,mouriglist,mouleflist,eyeopening2,broweyedif2]
                        # maxofmetricscen=[]
                        # minofmetricscen=[]
                        # for index,el in enumerate(metrics):
                        #         maxofmetricscen.append(np.max(el))
                        #         minofmetricscen.append(np.min(el))
                                               
                  #The following only need to be activated if we want to manually examine the maximum value of each metric
                  #and to create the tables with their values
                        
                        # print("\n")
                        # print("Below are metrics for central palsy")
                        
                        # print("eyeopening - Measure 8: Calculate difference in amount of eyelid opening between eyes (1)")
                        # print("Maximum for central")
                        # print(np.max(eyeopening))
                        # print("Minimum for central")
                        # print(np.min(eyeopening))
                        # print("Mean for central")
                        # print(np.mean(eyeopening))
                        # print("Std for central")
                        # print(np.std(eyeopening))
                        
                        # print("broweyedif - Measure 11: Difference between the difference of the") 
                        # print("distance of the lowest point of each eye to the highest of the corresponding brow")
                        # print("Maximum for central")
                        # print(np.max(broweyedif))
                        # print("Minimum for central")
                        # print(np.min(broweyedif))
                        # print("Mean for central")
                        # print(np.mean(broweyedif))
                        # print("Std for central")
                        # print(np.std(broweyedif))
                        
                        # print("eyeopening0 - Measure 12: Difference of distance between tip of nose and each brow")
                        # print("Maximum for central")
                        # print(np.max(eyeopening0))
                        # print("Minimum for central")
                        # print(np.min(eyeopening0))
                        # print("Mean for central")
                        # print(np.mean(eyeopening0))
                        # print("Std for central")
                        # print(np.std(eyeopening0))
                        
                        # print("eyeopening1 - Measure 13: Difference between most-left point of nose and each brow")
                        # print("Maximum for central")
                        # print(np.max(eyeopening1))
                        # print("Minimum for central")
                        # print(np.min(eyeopening1))
                        # print("Mean for central")
                        # print(np.mean(eyeopening1))
                        # print("Std for central")
                        # print(np.std(eyeopening1))
                        
                        # print("eyeopening3 - Measure 14: Distance between sides of mouth and corresponding brows")
                        # print("Maximum for central")
                        # print(np.max(eyeopening3))
                        # print("Minimum for central")
                        # print(np.min(eyeopening3))
                        # print("Mean for central")
                        # print(np.mean(eyeopening3))
                        # print("Std for central")
                        # print(np.std(eyeopening3))
                        
                        # # print("eyeopening6 - Measure 18: Difference of eyebrow heights")
                        # # print("Maximum for central")
                        # # print(np.max(eyeopening6))
                        # # print("Minimum for central")
                        # # print(np.min(eyeopening6))
                        # # print("Mean for central")
                        # # print(np.mean(eyeopening6))
                        # # print("Std for central")
                        # # print(np.std(eyeopening6))
                        
                        # print("broweyedif1 - Measure 19: Distance between lowest point on chin and almost top of nose")
                        # print("Maximum for central")
                        # print(np.max(broweyedif1))
                        # print("Minimum for central")
                        # print(np.min(broweyedif1))
                        # print("Mean for central")
                        # print(np.mean(broweyedif1))
                        # print("Std for central")
                        # print(np.std(broweyedif1))  
                        
                        # print("moutheyes5 - Measure 22: Calculate difference of opening of each side of the mouth")
                        # print("Maximum for central")
                        # print(np.max(moutheyes5))
                        # print("Minimum for central")
                        # print(np.min(moutheyes5))
                        # print("Mean for central")
                        # print(np.mean(moutheyes5))
                        # print("Std for central")
                        # print(np.std(moutheyes5))               
                        
                        # print("broweyedif2 - Sum all distances from the brow to mouth, nose, chin etc. - Added metric in loop")
                        # print("Maximum for central")
                        # print(np.max(broweyedif2))
                        # print("Minimum for central")
                        # print(np.min(broweyedif2))
                        # print("Mean for central")
                        # print(np.mean(broweyedif2))
                        # print("Std for central")
                        # print(np.std(broweyedif2))
                        
                        # # print("listbrowslopedif - Measure 17: Distance between lowest point on chin and brows")
                        # # print("Maximum for central")
                        # # print(np.max(listbrowslopedif))
                        # # print("Minimum for central")
                        # # print(np.min(listbrowslopedif))
                        # # print("Mean for central")
                        # # print(np.mean(listbrowslopedif))
                        # # print("Std for central")
                        # # print(np.std(listbrowslopedif))               
                        
                        # print("moutheyes0 - Measures 20: Differences of the average of 3 left from the average of 3 right points")
                        # print("on the top of each eye")
                        # print("Maximum for central")
                        # print(np.max(moutheyes0))
                        # print("Minimum for central")
                        # print(np.min(moutheyes0))
                        # print("Mean for central")
                        # print(np.mean(moutheyes0))
                        # print("Std for central")
                        # print(np.std(moutheyes0))
                        
                        
                        # print("eyeopening3b - Measure 16: Distance between sides of mouth and corresponding brows (try to amplify it)")
                        # print("Maximum for central")
                        # print(np.max(eyeopening3b))
                        # print("Minimum for central")
                        # print(np.min(eyeopening3b))
                        # print("Mean for central")
                        # print(np.mean(eyeopening3b))
                        # print("Std for central")
                        # print(np.std(eyeopening3b))   
                        
                        # print("eyeopeninga - Measures 9 and 10: A second and third way to calculate amount of eyelid opening")
                        # print("Maximum for central")
                        # print(np.max(eyeopeninga))
                        # print("Minimum for central")
                        # print(np.min(eyeopeninga))
                        # print("Mean for central")
                        # print(np.mean(eyeopeninga))
                        # print("Std for central")
                        # print(np.std(eyeopeninga))  
                        
                        # print("listrighta - Measures 23, 24, 25: Fit the best lines on left and right brows")
                        # print("Maximum for central")
                        # print(np.max(listrighta))
                        # print("Minimum for central")
                        # print(np.min(listrighta))
                        # print("Mean for central")
                        # print(np.mean(listrighta))
                        # print("Std for central")
                        # print(np.std(listrighta))
                        
                        # print("listline - Measure 34: Distance from top of nose to the farthest point of each eye ")
                        # print("Maximum for central")
                        # print(np.max(listline))
                        # print("Minimum for central")
                        # print(np.min(listline))
                        # print("Mean for central")
                        # print(np.mean(listline))
                        # print("Std for central")
                        # print(np.std(listline))
                        
                        # print("listlefte4 - Measures 46 and 47: Best line fit of the 2 most right bottom points of each eye") 
                        # print("and keep slopes")
                        # print("Maximum for central")
                        # print(np.max(listlefte4))
                        # print("Minimum for central")
                        # print(np.min(listlefte4))
                        # print("Mean for central")
                        # print(np.mean(listlefte4))
                        # print("Std for central")
                        # print(np.std(listlefte4))
                        
                        # print("listbroweye - Measure 49, 50, 51: Difference of the highest point of both brows")
                        # print("from the lowest of both brows")
                        # print("Maximum for central")
                        # print(np.max(listbroweye))
                        # print("Minimum for central")
                        # print(np.min(listbroweye))
                        # print("Mean for central")
                        # print(np.mean(listbroweye))
                        # print("Std for central")
                        # print(np.std(listbroweye))
                        
                        # print("eyeopening3a - Measure 15: Distance between sides of mouth and corresponding brows (try to amplify it)")
                        # print("Maximum for central")
                        # print(np.max(eyeopening3a))
                        # print("Minimum for central")
                        # print(np.min(eyeopening3a))
                        # print("Mean for central")
                        # print(np.mean(eyeopening3a))
                        # print("Std for central")
                        # print(np.std(eyeopening3a))
                        
                        
                        # print("listslrightleft - Measures 35, 36 and 37: Best line fit of 2 most left points of each eye")
                        # print("and keep slope and difference of slopes to a list")
                        # print("Maximum for central")
                        # print(np.max(listslrightleft))
                        # print("Minimum for central")
                        # print(np.min(listslrightleft))
                        # print("Mean for central")
                        # print(np.mean(listslrightleft))
                        # print("Std for central")
                        # print(np.std(listslrightleft))
                        
                        
                        # print("listmouthbrow - Measure 52: Difference of the highest from the lowest point of the 4 most-left")
                        # print("and 4 most-right points of the mouth")
                        # print("Maximum for central")
                        # print(np.max(listmouthbrow))
                        # print("Minimum for central")
                        # print(np.min(listmouthbrow))
                        # print("Mean for central")
                        # print(np.mean(listmouthbrow))
                        # print("Std for central")
                        # print(np.std(listmouthbrow))
                        
                        
                        # print("totaleyebrowdif - Measure 26: Calculate difference of top points in each brow and eye from each other")
                        # print("Maximum for central")
                        # print(np.max(totaleyebrowdif))
                        # print("Minimum for central")
                        # print(np.min(totaleyebrowdif))
                        # print("Mean for central")
                        # print(np.mean(totaleyebrowdif))
                        # print("Std for central")
                        # print(np.std(totaleyebrowdif))
                        
                        
                        # print("listbroweye2 - Measure 49, 50, 51: Difference of the highest point of both brows") 
                        # print("from the lowest of both brows")
                        # print("Maximum for central")
                        # print(np.max(listbroweye2))
                        # print("Minimum for central")
                        # print(np.min(listbroweye2))
                        # print("Mean for central")
                        # print(np.mean(listbroweye2))
                        # print("Std for central")
                        # print(np.std(listbroweye2))
                        
                        # print("listlefta - Measures 23, 24, 25: Fit the best lines on left and right brows")
                        # print("Maximum for central")
                        # print(np.max(listlefta))
                        # print("Minimum for central")
                        # print(np.min(listlefta))
                        # print("Mean for central")
                        # print(np.mean(listlefta))
                        # print("Std for central")
                        # print(np.std(listlefta))
                        
                        # print("listlefta1 - Measures 27 and 28: Best line fit of 3 most left points of each brow")
                        # print("and save the slope of each brow to a list")
                        # print("Maximum for central")
                        # print(np.max(listlefta1))
                        # print("Minimum for central")
                        # print(np.min(listlefta1))
                        # print("Mean for central")
                        # print(np.mean(listlefta1))
                        # print("Std for central")
                        # print(np.std(listlefta1))
                        
                        # print("listrighta1 - Measures 27 and 28: Best line fit of 3 most left points of each brow")
                        # print("and save the slope of each brow to a list")
                        # print("Maximum for central")
                        # print(np.max(listrighta1))
                        # print("Minimum for central")
                        # print(np.min(listrighta1))
                        # print("Mean for central")
                        # print(np.mean(listrighta1))
                        # print("Std for central")
                        # print(np.std(listrighta1))
                        
                        # print("listlefta2 - Measures 29 and 30: Best line fit of 3 middle points of each brow")
                        # print("and keep the slope of each one of them to a list")
                        # print("Maximum for central")
                        # print(np.max(listlefta2))
                        # print("Minimum for central")
                        # print(np.min(listlefta2))
                        # print("Mean for central")
                        # print(np.mean(listlefta2))
                        # print("Std for central")
                        # print(np.std(listlefta2))
                        
                        # print("listrighta2 - Measures 29 and 30: Best line fit of 3 middle points of each brow")
                        # print("and keep the slope of each one of them to a list")
                        # print("Maximum for central")
                        # print(np.max(listrighta2))
                        # print("Minimum for central")
                        # print(np.min(listrighta2))
                        # print("Mean for central")
                        # print(np.mean(listrighta2))
                        # print("Std for central")
                        # print(np.std(listrighta2))
                        
                        # print("listb - Measures 23, 24, 25: Fit the best lines on left and right brows")
                        # print("Maximum for central")
                        # print(np.max(listb))
                        # print("Minimum for central")
                        # print(np.min(listb))
                        # print("Mean for central")
                        # print(np.mean(listb))
                        # print("Std for central")
                        # print(np.std(listb))
                        
                        # print("listlefte - Measures 31, 32 and 33: Best line fit of 3 most left points of each eye")
                        # print("and keep the slope of each brow to a list")
                        # print("Maximum for central")
                        # print(np.max(listlefte))
                        # print("Minimum for central")
                        # print(np.min(listlefte))
                        # print("Mean for central")
                        # print(np.mean(listlefte))
                        # print("Std for central")
                        # print(np.std(listlefte))
                        
                        # print("listrighte - Measures 31, 32 and 33: Best line fit of 3 most left points of each eye")
                        # print("and keep the slope of each brow to a list")
                        # print("Maximum for central")
                        # print(np.max(listrighte))
                        # print("Minimum for central")
                        # print(np.min(listrighte))
                        # print("Mean for central")
                        # print(np.mean(listrighte))
                        # print("Std for central")
                        # print(np.std(listrighte))
                        
        
                        # print("listlefte1 - Measures 35, 36 and 37: Best line fit of 2 most left points of each eye")
                        # print("and keep slope and difference of slopes to a list")
                        # print("Maximum for central")
                        # print(np.max(listlefte1))
                        # print("Minimum for central")
                        # print(np.min(listlefte1))
                        # print("Mean for central")
                        # print(np.mean(listlefte1))
                        # print("Std for central")
                        # print(np.std(listlefte1))
                        
                        # print("listrighte1 - Measures 35, 36 and 37: Best line fit of 2 most left points of each eye")
                        # print("and keep slope and difference of slopes to a list")
                        # print("Maximum for central")
                        # print(np.max(listrighte1))
                        # print("Minimum for central")
                        # print(np.min(listrighte1))
                        # print("Mean for central")
                        # print(np.mean(listrighte1))
                        # print("Std for central")
                        # print(np.std(listrighte1))
                        
                        # print("listlefte2 - Measures 38 and 39: Best line fit of the 2 most right points of each eye")
                        # print("and keep slope in a list")
                        # print("Maximum for central")
                        # print(np.max(listlefte2))
                        # print("Minimum for central")
                        # print(np.min(listlefte2))
                        # print("Mean for central")
                        # print(np.mean(listlefte2))
                        # print("Std for central")
                        # print(np.std(listlefte2))
                        
                        # print("listrighte2 - Measures 38 and 39: Best line fit of the 2 most right points of each eye")
                        # print("and keep slope in a list")
                        # print("Maximum for central")
                        # print(np.max(listrighte2))
                        # print("Minimum for central")
                        # print(np.min(listrighte2))
                        # print("Mean for central")
                        # print(np.mean(listrighte2))
                        # print("Std for central")
                        # print(np.std(listrighte2))
                        
                        # print("listlefta3 - Measures 40 and 41: Best line fit of the 2 most left points of each brow")
                        # print("and keep slopes in a list")
                        # print("Maximum for central")
                        # print(np.max(listlefta3))
                        # print("Minimum for central")
                        # print(np.min(listlefta3))
                        # print("Mean for central")
                        # print(np.mean(listlefta3))
                        # print("Std for central")
                        # print(np.std(listlefta3))
                        
                        # print("listrighta3 - Measures 40 and 41: Best line fit of the 2 most left points of each brow")
                        # print("and keep slopes in a list")
                        # print("Maximum for central")
                        # print(np.max(listrighta3))
                        # print("Minimum for central")
                        # print(np.min(listrighta3))
                        # print("Mean for central")
                        # print(np.mean(listrighta3))
                        # print("Std for central")
                        # print(np.std(listrighta3))
                        
                        # print("listlefta4 - Measures 42 and 43: Best line fit of the 2 most right points of each brow")
                        # print("and keep slopes in a list")
                        # print("Maximum for central")
                        # print(np.max(listlefta4))
                        # print("Minimum for central")
                        # print(np.min(listlefta4))
                        # print("Mean for central")
                        # print(np.mean(listlefta4))
                        # print("Std for central")
                        # print(np.std(listlefta4))
                        
                        # print("listrighta4 - Measures 42 and 43: Best line fit of the 2 most right points of each brow")
                        # print("and keep slopes in a list")
                        # print("Maximum for central")
                        # print(np.max(listrighta4))
                        # print("Minimum for central")
                        # print(np.min(listrighta4))
                        # print("Mean for central")
                        # print(np.mean(listrighta4))
                        # print("Std for central")
                        # print(np.std(listrighta4))
                        
                        # print("listlefte3 - Measures 44 and 45: Best line fit of the 2 most left bottom points of each eye")
                        # print("and keep slopes")
                        # print("Maximum for central")
                        # print(np.max(listlefte3))
                        # print("Minimum for central")
                        # print(np.min(listlefte3))
                        # print("Mean for central")
                        # print(np.mean(listlefte3))
                        # print("Std for central")
                        # print(np.std(listlefte3))
                        
                        # print("listrighte3 - Measures 44 and 45: Best line fit of the 2 most left bottom points of each eye")
                        # print("and keep slopes")
                        # print("Maximum for central")
                        # print(np.max(listrighte3))
                        # print("Minimum for central")
                        # print(np.min(listrighte3))
                        # print("Mean for central")
                        # print(np.mean(listrighte3))
                        # print("Std for central")
                        # print(np.std(listrighte3))
                        
                        
                        # print("listrighte4 - Measures 46 and 47: Best line fit of the 2 most right bottom points of each eye")
                        # print("and keep slopes")
                        # print("Maximum for central")
                        # print(np.max(listrighte4))
                        # print("Minimum for central")
                        # print(np.min(listrighte4))
                        # print("Mean for central")
                        # print(np.mean(listrighte4))
                        # print("Std for central")
                        # print(np.std(listrighte4))
                        
                        # print("listeyedif - Measure 48: Difference of the highest point of both eyes from the lowest of both eyes")
                        # print("Maximum for central")
                        # print(np.max(listeyedif))
                        # print("Minimum for central")
                        # print(np.min(listeyedif))
                        # print("Mean for central")
                        # print(np.mean(listeyedif))
                        # print("Std for central")
                        # print(np.std(listeyedif))
                        
                        # print("listbrowdif - Measure 49, 50, 51: Difference of the highest point of both brows")
                        # print("from the lowest of both brows")
                        # print("Maximum for central")
                        # print(np.max(listbrowdif))
                        # print("Minimum for central")
                        # print(np.min(listbrowdif))
                        # print("Mean for central")
                        # print(np.mean(listbrowdif))
                        # print("Std for central")
                        # print(np.std(listbrowdif))
                        
                        # print("eyeopeningb - Measures 9 and 10: A second and third way to calculate amount of eyelid opening")
                        # print("Maximum for central")
                        # print(np.max(eyeopeningb))
                        # print("Minimum for central")
                        # print(np.min(eyeopeningb))
                        # print("Mean for central")
                        # print(np.mean(eyeopeningb))
                        # print("Std for central")
                        # print(np.std(eyeopeningb))
                        
        
                        
                        # print('\n')
                        # print('BELOW ARE LISTS THAT ARE MAINLY USED FOR DISTINGUISHING NORMAL/PATIENT')
                        
                        # print("moutheyes -  Measure 1: Distance from centre of eyes to edges of mouth")
                        # print("Maximum for central")
                        # print(np.max(moutheyes))
                        # print("Minimum for central")
                        # print(np.min(moutheyes))
                        # print("Mean for central")
                        # print(np.mean(moutheyes))
                        # print("Std for central")
                        # print(np.std(moutheyes))
                        
                        # print("eyesnose - Measure 2: Distance from centre of eyes to sides of the nose")
                        # print("Maximum for central")
                        # print(np.max(eyesnose))
                        # print("Minimum for central")
                        # print(np.min(eyesnose))
                        # print("Mean for central")
                        # print(np.mean(eyesnose))
                        # print("Std for central")
                        # print(np.std(eyesnose))
                        
                        # print("eyestopnose - Measure 3: Euclidean Distance of each eye from the top of the nose")
                        # print("Maximum for central")
                        # print(np.max(eyestopnose))
                        # print("Minimum for central")
                        # print(np.min(eyestopnose))
                        # print("Mean for central")
                        # print(np.mean(eyestopnose))
                        # print("Std for central")
                        # print(np.std(eyestopnose))
                        
                        # print("eyestopleftmouth - Measure 4: Euclidean Distance between each eye and the top left corner of the mouth")
                        # print("Maximum for central")
                        # print(np.max(eyestopleftmouth))
                        # print("Minimum for central")
                        # print(np.min(eyestopleftmouth))
                        # print("Mean for central")
                        # print(np.mean(eyestopleftmouth))
                        # print("Std for central")
                        # print(np.std(eyestopleftmouth))
                        
                        # print("eyesbottomrightmouth - Measure 5: Distance between eyes and bottom corner of the right side of the mouth")
                        # print("Maximum for central")
                        # print(np.max(eyesbottomrightmouth))
                        # print("Minimum for central")
                        # print(np.min(eyesbottomrightmouth))
                        # print("Mean for central")
                        # print(np.mean(eyesbottomrightmouth))
                        # print("Std for central")
                        # print(np.std(eyesbottomrightmouth))
                        
                        # print("topnosesidesofmouth - Measure 6: Difference in distance between top of nose and each side of the mouth")
                        # print("Maximum for central")
                        # print(np.max(topnosesidesofmouth))
                        # print("Minimum for central")
                        # print(np.min(topnosesidesofmouth))
                        # print("Mean for central")
                        # print(np.mean(topnosesidesofmouth))
                        # print("Std for central")
                        # print(np.std(topnosesidesofmouth))
                        
                        # print("eyeschinlow - Measure 7: Euclidean Distance between each eye and lowest point of the chin")
                        # print("Maximum for central")
                        # print(np.max(eyeschinlow))
                        # print("Minimum for central")
                        # print(np.min(eyeschinlow))
                        # print("Mean for central")
                        # print(np.mean(eyeschinlow))
                        # print("Std for central")
                        # print(np.std(eyeschinlow))  
                        
                        # print("eyeopening2 - Sum all distances from the eye to mouth, nose, chin etc. - Added in a loop")
                        # print("Maximum for central")
                        # print(np.max(eyeopening2))
                        # print("Minimum for central")
                        # print(np.min(eyeopening2))
                        # print("Mean for central")
                        # print(np.mean(eyeopening2))
                        # print("Std for central")
                        # print(np.std(eyeopening2))        
                        
                        
                        # print("newmet - Difference of left and right eye openings")
                        # print("Maximum for central")
                        # print(np.max(newmet))
                        # print("Minimum for central")
                        # print(np.min(newmet))
                        # print("Mean for central")
                        # print(np.mean(newmet))
                        # print("Std for central")
                        # print(np.std(newmet))   
                        
                        # print("newmet2 - Difference of left and right eyebrow heights")
                        # print("Maximum for central")
                        # print(np.max(newmet2))
                        # print("Minimum for central")
                        # print(np.min(newmet2))
                        # print("Mean for central")
                        # print(np.mean(newmet2))
                        # print("Std for central")
                        # print(np.std(newmet2))  
                        
                        # print("mouriglist - lowest point of right eye")
                        # print("Maximum for central")
                        # print(np.max(mouriglist))
                        # print("Minimum for central")
                        # print(np.min(mouriglist))
                        # print("Mean for central")
                        # print(np.mean(mouriglist))
                        # print("Std for central")
                        # print(np.std(mouriglist))  
                        
                        # print("mouleflist - lowest point of left eye")
                        # print("Maximum for central")
                        # print(np.max(mouleflist))
                        # print("Minimum for central")
                        # print(np.min(mouleflist))
                        # print("Mean for central")
                        # print(np.mean(mouleflist))
                        # print("Std for central")
                        # print(np.std(mouleflist))  
                        
                        # # print("ratioleft")
                        # # print("Maximum for central")
                        # # print(np.max(ratioleft))
                        # # print("Minimum for central")
                        # # print(np.min(ratioleft))
                        # # print("Mean for central")
                        # # print(np.mean(ratioleft))
                        # # print("Std for central")
                        # # print(np.std(ratioleft))  
                        
                        # # print("ratioright")
                        # # print("Maximum for central")
                        # # print(np.max(ratioright))
                        # # print("Minimum for central")
                        # # print(np.min(ratioright))
                        # # print("Mean for central")
                        # # print(np.mean(ratioright))
                        # # print("Std for central")
                        # # print(np.std(ratioright))  
                        
                        # print("lenratio - ratio of the two eye ratios")
                        # print("Maximum for central")
                        # print(np.max(lenratio))
                        # print("Minimum for central")
                        # print(np.min(lenratio))
                        # print("Mean for central")
                        # print(np.mean(lenratio))
                        # print("Std for central")
                        # print(np.std(lenratio))  
                        
        
                 #Below there are metrics used to distinguish peripheral from central palsy       
                        print('\n')
                        fpfinlis1=[] #initialize empty list to be used below
                
                        fpfin1=0 #False Positives to 0
                        tnfin1=0 #True Negatives to 0
                        for i in range(len(eyeopening)): #Loop over the number of central images
                            if int(manual_an)==1: #Below are the metrics used when we have manual annotation
                                if any([eval(metrnamespat[u]) for u in range(p+1)])==True: 
                                    fpfin1=fpfin1+1
                                    fpfinlis1.append(listofnames[i]) #Append the name of misclassified image to a list                       
                                else:
                                      tnfin1=tnfin1+1
                            else: #Below are the metrics used when we have automatic annotation                  
                                if any([eval(metrnamespat2[u]) for u in range(p+1)])==True:
                                    fpfin1=fpfin1+1
                                    fpfinlis1.append(listofnames[i]) #Append the name of misclassified image to a list 
                                else:
                                      tnfin1=tnfin1+1               
                        fnfinlis.append('path2-central from now on') #Because we have already added the misclassified peripherals to that list
        
                    for i in range(len(eyeopening2)): #Loop over the total number of patient images
                        #If one of the below metrics is true then the image is correctly classified as patient
                        if int(manual_an)==1: #These are the metrics for manual annotation
                            if any([eval(metrnames[k]) for k in range(l+1)])==True:
                                tpfin=tpfin+1
                            else: #If all the below metrics are false then the image is incorrectly classified as healthy
                                fnfin=fnfin+1
                                fnfinlis.append(listofnames[i]) #Append the name of the incorrectly classified image to a list
                        else: #These are the metrics for automatic annotation
                            if any([eval(metrnames2[k]) for k in range(l+1)])==True:
                                tpfin=tpfin+1
                            else: #If all the below metrics are false then the image is incorrectly classified as healthy
                                fnfin=fnfin+1
                                fnfinlis.append(listofnames[i]) #Append the name of the incorrectly classified image to a list
        
        
        
                elif path==pathheal: #Below metrics for healthy individuals are printed
                    
                    #The following only need to be activated if we want to use the exact form of algorithm2
                    # metrics=[eyeopening,broweyedif,moutheyes,eyesnose,eyestopnose,eyestopleftmouth,eyesbottomrightmouth,topnosesidesofmouth,
                    #       eyeopening0,eyeopening1,eyeopening3,eyeschinlow,broweyedif1,moutheyes0,moutheyes5,eyeopening3a,eyeopening3b,
                    #       eyeopeninga,listlefta,listb,listrighta,listlefta1,listrighta1,listlefta2,listrighta2,eyeopeningb,listlefte,listrighte,
                    #       listline,listlefte1,listrighte1,listlefte2,listrighte2,listlefta3,listrighta3,listlefta4,listrighta4,listlefte3,listrighte3,
                    #       listlefte4,listrighte4,listeyedif,listbrowdif,listbroweye,listslrightleft,listmouthbrow,totaleyebrowdif,listbroweye2,
                    #       lenratio,newmet,newmet2,mouriglist,mouleflist,eyeopening2,broweyedif2]
                    # maxofmetricshe=[]
                    # minofmetricshe=[]
                    # for index,el in enumerate(metrics):
                    #         maxofmetricshe.append(np.max(el))
                    #         minofmetricshe.append(np.min(el))
              
            #The following only need to be activated if we want to manually examine the maximum value of each metric
            #and to create the tables with their values
                            
                    # print("\n")
                    # print("Below are metrics for normal")
                    
                    # print("eyeopening - Measure 8: Calculate difference in amount of eyelid opening between eyes (1)")
                    # print("Maximum for normal")
                    # print(np.max(eyeopening))
                    # print("Minimum for normal")
                    # print(np.min(eyeopening))
                    # print("Mean for normal")
                    # print(np.mean(eyeopening))
                    # print("Std for normal")
                    # print(np.std(eyeopening))
                    
                    # print("broweyedif - Measure 11: Difference between the difference of the") 
                    # print("distance of the lowest point of each eye to the highest of the corresponding brow")
                    # print("Maximum for normal")
                    # print(np.max(broweyedif))
                    # print("Minimum for normal")
                    # print(np.min(broweyedif))
                    # print("Mean for normal")
                    # print(np.mean(broweyedif))
                    # print("Std for normal")
                    # print(np.std(broweyedif))
                    
                    # print("eyeopening0 - Measure 12: Difference of distance between tip of nose and each brow")
                    # print("Maximum for normal")
                    # print(np.max(eyeopening0))
                    # print("Minimum for normal")
                    # print(np.min(eyeopening0))
                    # print("Mean for normal")
                    # print(np.mean(eyeopening0))
                    # print("Std for normal")
                    # print(np.std(eyeopening0))
                    
                    # print("eyeopening1 - Measure 13: Difference between most-left point of nose and each brow")
                    # print("Maximum for normal")
                    # print(np.max(eyeopening1))
                    # print("Minimum for normal")
                    # print(np.min(eyeopening1))
                    # print("Mean for normal")
                    # print(np.mean(eyeopening1))
                    # print("Std for normal")
                    # print(np.std(eyeopening1))
                    
                    # print("eyeopening3 - Measure 14: Distance between sides of mouth and corresponding brows")
                    # print("Maximum for normal")
                    # print(np.max(eyeopening3))
                    # print("Minimum for normal")
                    # print(np.min(eyeopening3))
                    # print("Mean for normal")
                    # print(np.mean(eyeopening3))
                    # print("Std for normal")
                    # print(np.std(eyeopening3))
                    
                    # # print("eyeopening6 - Measure 18: Difference of eyebrow heights")
                    # # print("Maximum for normal")
                    # # print(np.max(eyeopening6))
                    # # print("Minimum for normal")
                    # # print(np.min(eyeopening6))
                    # # print("Mean for normal")
                    # # print(np.mean(eyeopening6))
                    # # print("Std for normal")
                    # # print(np.std(eyeopening6))
                    
                    # print("broweyedif1 - Measure 19: Distance between lowest point on chin and almost top of nose")
                    # print("Maximum for normal")
                    # print(np.max(broweyedif1))
                    # print("Minimum for normal")
                    # print(np.min(broweyedif1))
                    # print("Mean for normal")
                    # print(np.mean(broweyedif1))
                    # print("Std for normal")
                    # print(np.std(broweyedif1))  
                    
                    # print("moutheyes5 - Measure 22: Calculate difference of opening of each side of the mouth")
                    # print("Maximum for normal")
                    # print(np.max(moutheyes5))
                    # print("Minimum for normal")
                    # print(np.min(moutheyes5))
                    # print("Mean for normal")
                    # print(np.mean(moutheyes5))
                    # print("Std for normal")
                    # print(np.std(moutheyes5))               
                    
                    # print("broweyedif2 - Sum all distances from the brow to mouth, nose, chin etc. - Added metric in loop")
                    # print("Maximum for normal")
                    # print(np.max(broweyedif2))
                    # print("Minimum for normal")
                    # print(np.min(broweyedif2))
                    # print("Mean for normal")
                    # print(np.mean(broweyedif2))
                    # print("Std for normal")
                    # print(np.std(broweyedif2))
                    
                    # # print("listbrowslopedif - Measure 17: Distance between lowest point on chin and brows")
                    # # print("Maximum for normal")
                    # # print(np.max(listbrowslopedif))
                    # # print("Minimum for normal")
                    # # print(np.min(listbrowslopedif))
                    # # print("Mean for normal")
                    # # print(np.mean(listbrowslopedif))
                    # # print("Std for normal")
                    # # print(np.std(listbrowslopedif))               
                    
                    # print("moutheyes0 - Measures 20: Differences of the average of 3 left from the average of 3 right points")
                    # print("on the top of each eye")
                    # print("Maximum for normal")
                    # print(np.max(moutheyes0))
                    # print("Minimum for normal")
                    # print(np.min(moutheyes0))
                    # print("Mean for normal")
                    # print(np.mean(moutheyes0))
                    # print("Std for normal")
                    # print(np.std(moutheyes0))
                    
                    
                    # print("eyeopening3b - Measure 16: Distance between sides of mouth and corresponding brows (try to amplify it)")
                    # print("Maximum for normal")
                    # print(np.max(eyeopening3b))
                    # print("Minimum for normal")
                    # print(np.min(eyeopening3b))
                    # print("Mean for normal")
                    # print(np.mean(eyeopening3b))
                    # print("Std for normal")
                    # print(np.std(eyeopening3b))   
                    
                    # print("eyeopeninga - Measures 9 and 10: A second and third way to calculate amount of eyelid opening")
                    # print("Maximum for normal")
                    # print(np.max(eyeopeninga))
                    # print("Minimum for normal")
                    # print(np.min(eyeopeninga))
                    # print("Mean for normal")
                    # print(np.mean(eyeopeninga))
                    # print("Std for normal")
                    # print(np.std(eyeopeninga))  
                    
                    # print("listrighta - Measures 23, 24, 25: Fit the best lines on left and right brows")
                    # print("Maximum for normal")
                    # print(np.max(listrighta))
                    # print("Minimum for normal")
                    # print(np.min(listrighta))
                    # print("Mean for normal")
                    # print(np.mean(listrighta))
                    # print("Std for normal")
                    # print(np.std(listrighta))
                    
                    # print("listline - Measure 34: Distance from top of nose to the farthest point of each eye ")
                    # print("Maximum for normal")
                    # print(np.max(listline))
                    # print("Minimum for normal")
                    # print(np.min(listline))
                    # print("Mean for normal")
                    # print(np.mean(listline))
                    # print("Std for normal")
                    # print(np.std(listline))
                    
                    # print("listlefte4 - Measures 46 and 47: Best line fit of the 2 most right bottom points of each eye") 
                    # print("and keep slopes")
                    # print("Maximum for normal")
                    # print(np.max(listlefte4))
                    # print("Minimum for normal")
                    # print(np.min(listlefte4))
                    # print("Mean for normal")
                    # print(np.mean(listlefte4))
                    # print("Std for normal")
                    # print(np.std(listlefte4))
                    
                    # print("listbroweye - Measure 49, 50, 51: Difference of the highest point of both brows")
                    # print("from the lowest of both brows")
                    # print("Maximum for normal")
                    # print(np.max(listbroweye))
                    # print("Minimum for normal")
                    # print(np.min(listbroweye))
                    # print("Mean for normal")
                    # print(np.mean(listbroweye))
                    # print("Std for normal")
                    # print(np.std(listbroweye))
                    
                    # print("eyeopening3a - Measure 15: Distance between sides of mouth and corresponding brows (try to amplify it)")
                    # print("Maximum for normal")
                    # print(np.max(eyeopening3a))
                    # print("Minimum for normal")
                    # print(np.min(eyeopening3a))
                    # print("Mean for normal")
                    # print(np.mean(eyeopening3a))
                    # print("Std for normal")
                    # print(np.std(eyeopening3a))
                    
                    
                    # print("listslrightleft - Measures 35, 36 and 37: Best line fit of 2 most left points of each eye")
                    # print("and keep slope and difference of slopes to a list")
                    # print("Maximum for normal")
                    # print(np.max(listslrightleft))
                    # print("Minimum for normal")
                    # print(np.min(listslrightleft))
                    # print("Mean for normal")
                    # print(np.mean(listslrightleft))
                    # print("Std for normal")
                    # print(np.std(listslrightleft))
                    
                    
                    # print("listmouthbrow - Measure 52: Difference of the highest from the lowest point of the 4 most-left")
                    # print("and 4 most-right points of the mouth")
                    # print("Maximum for normal")
                    # print(np.max(listmouthbrow))
                    # print("Minimum for normal")
                    # print(np.min(listmouthbrow))
                    # print("Mean for normal")
                    # print(np.mean(listmouthbrow))
                    # print("Std for normal")
                    # print(np.std(listmouthbrow))
                    
                    
                    # print("totaleyebrowdif - Measure 26: Calculate difference of top points in each brow and eye from each other")
                    # print("Maximum for normal")
                    # print(np.max(totaleyebrowdif))
                    # print("Minimum for normal")
                    # print(np.min(totaleyebrowdif))
                    # print("Mean for normal")
                    # print(np.mean(totaleyebrowdif))
                    # print("Std for normal")
                    # print(np.std(totaleyebrowdif))
                    
                    
                    # print("listbroweye2 - Measure 49, 50, 51: Difference of the highest point of both brows") 
                    # print("from the lowest of both brows")
                    # print("Maximum for normal")
                    # print(np.max(listbroweye2))
                    # print("Minimum for normal")
                    # print(np.min(listbroweye2))
                    # print("Mean for normal")
                    # print(np.mean(listbroweye2))
                    # print("Std for normal")
                    # print(np.std(listbroweye2))
                    
                    # print("listlefta - Measures 23, 24, 25: Fit the best lines on left and right brows")
                    # print("Maximum for normal")
                    # print(np.max(listlefta))
                    # print("Minimum for normal")
                    # print(np.min(listlefta))
                    # print("Mean for normal")
                    # print(np.mean(listlefta))
                    # print("Std for normal")
                    # print(np.std(listlefta))
                    
                    # print("listlefta1 - Measures 27 and 28: Best line fit of 3 most left points of each brow")
                    # print("and save the slope of each brow to a list")
                    # print("Maximum for normal")
                    # print(np.max(listlefta1))
                    # print("Minimum for normal")
                    # print(np.min(listlefta1))
                    # print("Mean for normal")
                    # print(np.mean(listlefta1))
                    # print("Std for normal")
                    # print(np.std(listlefta1))
                    
                    # print("listrighta1 - Measures 27 and 28: Best line fit of 3 most left points of each brow")
                    # print("and save the slope of each brow to a list")
                    # print("Maximum for normal")
                    # print(np.max(listrighta1))
                    # print("Minimum for normal")
                    # print(np.min(listrighta1))
                    # print("Mean for normal")
                    # print(np.mean(listrighta1))
                    # print("Std for normal")
                    # print(np.std(listrighta1))
                    
                    # print("listlefta2 - Measures 29 and 30: Best line fit of 3 middle points of each brow")
                    # print("and keep the slope of each one of them to a list")
                    # print("Maximum for normal")
                    # print(np.max(listlefta2))
                    # print("Minimum for normal")
                    # print(np.min(listlefta2))
                    # print("Mean for normal")
                    # print(np.mean(listlefta2))
                    # print("Std for normal")
                    # print(np.std(listlefta2))
                    
                    # print("listrighta2 - Measures 29 and 30: Best line fit of 3 middle points of each brow")
                    # print("and keep the slope of each one of them to a list")
                    # print("Maximum for normal")
                    # print(np.max(listrighta2))
                    # print("Minimum for normal")
                    # print(np.min(listrighta2))
                    # print("Mean for normal")
                    # print(np.mean(listrighta2))
                    # print("Std for normal")
                    # print(np.std(listrighta2))
                    
                    # print("listb - Measures 23, 24, 25: Fit the best lines on left and right brows")
                    # print("Maximum for normal")
                    # print(np.max(listb))
                    # print("Minimum for normal")
                    # print(np.min(listb))
                    # print("Mean for normal")
                    # print(np.mean(listb))
                    # print("Std for normal")
                    # print(np.std(listb))
                    
                    # print("listlefte - Measures 31, 32 and 33: Best line fit of 3 most left points of each eye")
                    # print("and keep the slope of each brow to a list")
                    # print("Maximum for normal")
                    # print(np.max(listlefte))
                    # print("Minimum for normal")
                    # print(np.min(listlefte))
                    # print("Mean for normal")
                    # print(np.mean(listlefte))
                    # print("Std for normal")
                    # print(np.std(listlefte))
                    
                    # print("listrighte - Measures 31, 32 and 33: Best line fit of 3 most left points of each eye")
                    # print("and keep the slope of each brow to a list")
                    # print("Maximum for normal")
                    # print(np.max(listrighte))
                    # print("Minimum for normal")
                    # print(np.min(listrighte))
                    # print("Mean for normal")
                    # print(np.mean(listrighte))
                    # print("Std for normal")
                    # print(np.std(listrighte))
                    
        
                    # print("listlefte1 - Measures 35, 36 and 37: Best line fit of 2 most left points of each eye")
                    # print("and keep slope and difference of slopes to a list")
                    # print("Maximum for normal")
                    # print(np.max(listlefte1))
                    # print("Minimum for normal")
                    # print(np.min(listlefte1))
                    # print("Mean for normal")
                    # print(np.mean(listlefte1))
                    # print("Std for normal")
                    # print(np.std(listlefte1))
                    
                    # print("listrighte1 - Measures 35, 36 and 37: Best line fit of 2 most left points of each eye")
                    # print("and keep slope and difference of slopes to a list")
                    # print("Maximum for normal")
                    # print(np.max(listrighte1))
                    # print("Minimum for normal")
                    # print(np.min(listrighte1))
                    # print("Mean for normal")
                    # print(np.mean(listrighte1))
                    # print("Std for normal")
                    # print(np.std(listrighte1))
                    
                    # print("listlefte2 - Measures 38 and 39: Best line fit of the 2 most right points of each eye")
                    # print("and keep slope in a list")
                    # print("Maximum for normal")
                    # print(np.max(listlefte2))
                    # print("Minimum for normal")
                    # print(np.min(listlefte2))
                    # print("Mean for normal")
                    # print(np.mean(listlefte2))
                    # print("Std for normal")
                    # print(np.std(listlefte2))
                    
                    # print("listrighte2 - Measures 38 and 39: Best line fit of the 2 most right points of each eye")
                    # print("and keep slope in a list")
                    # print("Maximum for normal")
                    # print(np.max(listrighte2))
                    # print("Minimum for normal")
                    # print(np.min(listrighte2))
                    # print("Mean for normal")
                    # print(np.mean(listrighte2))
                    # print("Std for normal")
                    # print(np.std(listrighte2))
                    
                    # print("listlefta3 - Measures 40 and 41: Best line fit of the 2 most left points of each brow")
                    # print("and keep slopes in a list")
                    # print("Maximum for normal")
                    # print(np.max(listlefta3))
                    # print("Minimum for normal")
                    # print(np.min(listlefta3))
                    # print("Mean for normal")
                    # print(np.mean(listlefta3))
                    # print("Std for normal")
                    # print(np.std(listlefta3))
                    
                    # print("listrighta3 - Measures 40 and 41: Best line fit of the 2 most left points of each brow")
                    # print("and keep slopes in a list")
                    # print("Maximum for normal")
                    # print(np.max(listrighta3))
                    # print("Minimum for normal")
                    # print(np.min(listrighta3))
                    # print("Mean for normal")
                    # print(np.mean(listrighta3))
                    # print("Std for normal")
                    # print(np.std(listrighta3))
                    
                    # print("listlefta4 - Measures 42 and 43: Best line fit of the 2 most right points of each brow")
                    # print("and keep slopes in a list")
                    # print("Maximum for normal")
                    # print(np.max(listlefta4))
                    # print("Minimum for normal")
                    # print(np.min(listlefta4))
                    # print("Mean for normal")
                    # print(np.mean(listlefta4))
                    # print("Std for normal")
                    # print(np.std(listlefta4))
                    
                    # print("listrighta4 - Measures 42 and 43: Best line fit of the 2 most right points of each brow")
                    # print("and keep slopes in a list")
                    # print("Maximum for normal")
                    # print(np.max(listrighta4))
                    # print("Minimum for normal")
                    # print(np.min(listrighta4))
                    # print("Mean for normal")
                    # print(np.mean(listrighta4))
                    # print("Std for normal")
                    # print(np.std(listrighta4))
                    
                    # print("listlefte3 - Measures 44 and 45: Best line fit of the 2 most left bottom points of each eye")
                    # print("and keep slopes")
                    # print("Maximum for normal")
                    # print(np.max(listlefte3))
                    # print("Minimum for normal")
                    # print(np.min(listlefte3))
                    # print("Mean for normal")
                    # print(np.mean(listlefte3))
                    # print("Std for normal")
                    # print(np.std(listlefte3))
                    
                    # print("listrighte3 - Measures 44 and 45: Best line fit of the 2 most left bottom points of each eye")
                    # print("and keep slopes")
                    # print("Maximum for normal")
                    # print(np.max(listrighte3))
                    # print("Minimum for normal")
                    # print(np.min(listrighte3))
                    # print("Mean for normal")
                    # print(np.mean(listrighte3))
                    # print("Std for normal")
                    # print(np.std(listrighte3))
                    
                    
                    # print("listrighte4 - Measures 46 and 47: Best line fit of the 2 most right bottom points of each eye")
                    # print("and keep slopes")
                    # print("Maximum for normal")
                    # print(np.max(listrighte4))
                    # print("Minimum for normal")
                    # print(np.min(listrighte4))
                    # print("Mean for normal")
                    # print(np.mean(listrighte4))
                    # print("Std for normal")
                    # print(np.std(listrighte4))
                    
                    # print("listeyedif - Measure 48: Difference of the highest point of both eyes from the lowest of both eyes")
                    # print("Maximum for normal")
                    # print(np.max(listeyedif))
                    # print("Minimum for normal")
                    # print(np.min(listeyedif))
                    # print("Mean for normal")
                    # print(np.mean(listeyedif))
                    # print("Std for normal")
                    # print(np.std(listeyedif))
                    
                    # print("listbrowdif - Measure 49, 50, 51: Difference of the highest point of both brows")
                    # print("from the lowest of both brows")
                    # print("Maximum for normal")
                    # print(np.max(listbrowdif))
                    # print("Minimum for normal")
                    # print(np.min(listbrowdif))
                    # print("Mean for normal")
                    # print(np.mean(listbrowdif))
                    # print("Std for normal")
                    # print(np.std(listbrowdif))
                    
                    # print("eyeopeningb - Measures 9 and 10: A second and third way to calculate amount of eyelid opening")
                    # print("Maximum for normal")
                    # print(np.max(eyeopeningb))
                    # print("Minimum for normal")
                    # print(np.min(eyeopeningb))
                    # print("Mean for normal")
                    # print(np.mean(eyeopeningb))
                    # print("Std for normal")
                    # print(np.std(eyeopeningb))
                    
        
                    
                    # print('\n')
                    # print('BELOW ARE LISTS THAR ARE MAINLY USED FOR DISTINGUISHING NORMAL/PATIENT')
                    
                    # print("moutheyes -  Measure 1: Distance from centre of eyes to edges of mouth")
                    # print("Maximum for normal")
                    # print(np.max(moutheyes))
                    # print("Minimum for normal")
                    # print(np.min(moutheyes))
                    # print("Mean for normal")
                    # print(np.mean(moutheyes))
                    # print("Std for normal")
                    # print(np.std(moutheyes))
                    
                    # print("eyesnose - Measure 2: Distance from centre of eyes to sides of the nose")
                    # print("Maximum for normal")
                    # print(np.max(eyesnose))
                    # print("Minimum for normal")
                    # print(np.min(eyesnose))
                    # print("Mean for normal")
                    # print(np.mean(eyesnose))
                    # print("Std for normal")
                    # print(np.std(eyesnose))
                    
                    # print("eyestopnose - Measure 3: Euclidean Distance of each eye from the top of the nose")
                    # print("Maximum for normal")
                    # print(np.max(eyestopnose))
                    # print("Minimum for normal")
                    # print(np.min(eyestopnose))
                    # print("Mean for normal")
                    # print(np.mean(eyestopnose))
                    # print("Std for normal")
                    # print(np.std(eyestopnose))
                    
                    # print("eyestopleftmouth - Measure 4: Euclidean Distance between each eye and the top left corner of the mouth")
                    # print("Maximum for normal")
                    # print(np.max(eyestopleftmouth))
                    # print("Minimum for normal")
                    # print(np.min(eyestopleftmouth))
                    # print("Mean for normal")
                    # print(np.mean(eyestopleftmouth))
                    # print("Std for normal")
                    # print(np.std(eyestopleftmouth))
                    
                    # print("eyesbottomrightmouth - Measure 5: Distance between eyes and bottom corner of the right side of the mouth")
                    # print("Maximum for normal")
                    # print(np.max(eyesbottomrightmouth))
                    # print("Minimum for normal")
                    # print(np.min(eyesbottomrightmouth))
                    # print("Mean for normal")
                    # print(np.mean(eyesbottomrightmouth))
                    # print("Std for normal")
                    # print(np.std(eyesbottomrightmouth))
                    
                    # print("topnosesidesofmouth - Measure 6: Difference in distance between top of nose and each side of the mouth")
                    # print("Maximum for normal")
                    # print(np.max(topnosesidesofmouth))
                    # print("Minimum for normal")
                    # print(np.min(topnosesidesofmouth))
                    # print("Mean for normal")
                    # print(np.mean(topnosesidesofmouth))
                    # print("Std for normal")
                    # print(np.std(topnosesidesofmouth))
                    
                    # print("eyeschinlow - Measure 7: Euclidean Distance between each eye and lowest point of the chin")
                    # print("Maximum for normal")
                    # print(np.max(eyeschinlow))
                    # print("Minimum for normal")
                    # print(np.min(eyeschinlow))
                    # print("Mean for normal")
                    # print(np.mean(eyeschinlow))
                    # print("Std for normal")
                    # print(np.std(eyeschinlow))  
                    
                    # print("eyeopening2 - Sum all distances from the eye to mouth, nose, chin etc. - Added in a loop")
                    # print("Maximum for normal")
                    # print(np.max(eyeopening2))
                    # print("Minimum for normal")
                    # print(np.min(eyeopening2))
                    # print("Mean for normal")
                    # print(np.mean(eyeopening2))
                    # print("Std for normal")
                    # print(np.std(eyeopening2))     
                    
                        
                    # print("newmet - Difference of left and right eye openings")
                    # print("Maximum for normal")
                    # print(np.max(newmet))
                    # print("Minimum for normal")
                    # print(np.min(newmet))
                    # print("Mean for normal")
                    # print(np.mean(newmet))
                    # print("Std for normal")
                    # print(np.std(newmet))   
                    
                    # print("newmet2 - Difference of left and right eyebrow heights")
                    # print("Maximum for normal")
                    # print(np.max(newmet2))
                    # print("Minimum for normal")
                    # print(np.min(newmet2))
                    # print("Mean for normal")
                    # print(np.mean(newmet2))
                    # print("Std for normal")
                    # print(np.std(newmet2))  
                    
                    # print("mouriglist - lowest point of right eye")
                    # print("Maximum for normal")
                    # print(np.max(mouriglist))
                    # print("Minimum for normal")
                    # print(np.min(mouriglist))
                    # print("Mean for normal")
                    # print(np.mean(mouriglist))
                    # print("Std for normal")
                    # print(np.std(mouriglist))  
                    
                    # print("mouleflist - lowest point of left eye")
                    # print("Maximum for normal")
                    # print(np.max(mouleflist))
                    # print("Minimum for normal")
                    # print(np.min(mouleflist))
                    # print("Mean for normal")
                    # print(np.mean(mouleflist))
                    # print("Std for normal")
                    # print(np.std(mouleflist))  
                    
                    # # print("ratioleft")
                    # # print("Maximum for normal")
                    # # print(np.max(ratioleft))
                    # # print("Minimum for normal")
                    # # print(np.min(ratioleft))
                    # # print("Mean for normal")
                    # # print(np.mean(ratioleft))
                    # # print("Std for normal")
                    # # print(np.std(ratioleft))  
                    
                    # # print("ratioright")
                    # # print("Maximum for normal")
                    # # print(np.max(ratioright))
                    # # print("Minimum for normal")
                    # # print(np.min(ratioright))
                    # # print("Mean for normal")
                    # # print(np.mean(ratioright))
                    # # print("Std for normal")
                    # # print(np.std(ratioright))  
                    
                    # print("lenratio - ratio of the two eye ratios")
                    # print("Maximum for normal")
                    # print(np.max(lenratio))
                    # print("Minimum for normal")
                    # print(np.min(lenratio))
                    # print("Mean for normal")
                    # print(np.mean(lenratio))
                    # print("Std for normal")
                    # print(np.std(lenratio))  
                    
                    #An empty list is initialized and fp and tn are set to 0
                    fpfinlis=[]
                    fpfin=0
                    tnfin=0
                    for i in range(len(eyeopening2)): #Loop over files of healthy individuals
                        #If one of the following metrics is true then the image is incorretly classified as patient
                        if int(manual_an)==1:
                            if any([eval(metrnames[k]) for k in range(l+1)])==True: 
                                fpfin=fpfin+1 #Increase false positives by 1 if one of the above conditions holds true
                                fpfinlis.append(listofnames[i]) #Add the false positive name to a list
                            else: #If all of the above metrics are false then the image is corretly classified as healthy
                                tnfin=tnfin+1 
                        else:
                            if any([eval(metrnames2[k]) for k in range(l+1)])==True:
                                fpfin=fpfin+1 #Increase false positives by 1 if one of the above conditions holds true
                                fpfinlis.append(listofnames[i]) #Add the false positive name to a list
                            else: #If all of the above metrics are false then the image is corretly classified as healthy
                                tnfin=tnfin+1 
        
                    #Append the lists of fp and tn
                    fplastfin2.append(fpfin)
                    tnlastfin2.append(tnfin)
        
                            
        #############################################
#Below algorithm2 actually calculate the thresholds. Only to be activated if needed
                    
        #Threshold Selection
        # thrnormhemax=[]
        # thrnormpatmax=[]
        # thrnormhemin=[]
        # thrnormpatmin=[]
        
        # thrpatcenmax=[]
        # thrpatpermax=[]
        # thrpatcenmin=[]
        # thrpatpermin=[]
        
        # thrnormhemaxlist=[]
        # thrnormpatmaxlist=[]
        # thrnormheminlist=[]
        # thrnormpatminlist=[]
        
        # thrpatcenmaxlist=[]
        # thrpatpermaxlist=[]
        # thrpatcenminlist=[]
        # thrpatperminlist=[]
        
        # for i in range(len(metrics)):
        #     #distiguishing healthy from patients (perfectly identify patients based on max)
        #     if maxofmetricshe[i]>np.maximum(maxofmetricsper[i],maxofmetricscen[i]):
        #         thrnormhemax.append((maxofmetricshe[i]+np.maximum(maxofmetricsper[i],maxofmetricscen[i]))/2) #Above that healthy
        #         thrnormhemaxlist.append(metricslist[i]+'>')
        #     #distiguishing healthy from patients (perfectly identify healthy based on max)
        #     if maxofmetricshe[i]<np.minimum(maxofmetricsper[i],maxofmetricscen[i]):
        #         thrnormpatmax.append((maxofmetricshe[i]+np.minimum(maxofmetricsper[i],maxofmetricscen[i]))/2) #Above that patient
        #         thrnormpatmaxlist.append(metricslist[i]+'>')
        #     #distiguishing healthy from patients (perfectly identify patients based on min)
        #     if minofmetricshe[i]>np.maximum(minofmetricsper[i],minofmetricscen[i]):
        #         thrnormhemin.append((minofmetricshe[i]+np.maximum(minofmetricsper[i],minofmetricscen[i]))/2) #below that patient
        #         thrnormheminlist.append(metricslist[i]+'<')
        #     #distiguishing healthy from patients (perfectly identify healthy based on min)
        #     if minofmetricshe[i]<np.minimum(minofmetricsper[i],minofmetricscen[i]):
        #         thrnormpatmin.append((minofmetricshe[i]+np.minimum(minofmetricsper[i],minofmetricscen[i]))/2) #below that healthy
        #         thrnormpatminlist.append(metricslist[i]+'<')
        
        #     #distiguishing central from peripheral (perfectly identify central based on max)
        #     if maxofmetricsper[i]>maxofmetricscen[i]:
        #         thrpatpermax.append((maxofmetricsper[i]+maxofmetricscen[i])/2) #Above that peripheral
        #         thrpatpermaxlist.append(metricslist[i]+'>')
        #     #distiguishing central from peripheral (perfectly identify peripheral based on max)
        #     if maxofmetricsper[i]<maxofmetricscen[i]:
        #         thrpatcenmax.append((maxofmetricsper[i]+maxofmetricscen[i])/2) #Above that central
        #         thrpatcenmaxlist.append(metricslist[i]+'>')
        #     #distiguishing central from peripheral (perfectly identify central based on min)
        #     if minofmetricsper[i]>minofmetricscen[i]:
        #         thrpatcenmin.append((minofmetricsper[i]+minofmetricscen[i])/2) #below that central
        #         thrpatcenminlist.append(metricslist[i]+'<')
        #     #distiguishing central from peripheral (perfectly identify peripheral based on min)
        #     if minofmetricsper[i]<minofmetricscen[i]:
        #         thrpatpermin.append((minofmetricsper[i]+minofmetricscen[i])/2) #below that peripheral
        #         thrpatperminlist.append(metricslist[i]+'<')
        
        
        # print("#distiguishing healthy from patients (perfectly identify patients based on max) #Above that healthy")
        # print(thrnormhemaxlist)
        # print(thrnormhemax)
        # print("#distiguishing healthy from patients (perfectly identify healthy based on max) #Above that patient")
        # print(thrnormpatmaxlist)
        # print(thrnormpatmax)
        # print("#distiguishing healthy from patients (perfectly identify patients based on min) #below that patient")
        # print(thrnormheminlist)
        # print(thrnormhemin)
        # print("#distiguishing healthy from patients (perfectly identify healthy based on min) #below that healthy")
        # print(thrnormpatminlist)
        # print(thrnormpatmin)
        
        # print("#distiguishing central from peripheral (perfectly identify central based on max) #Above that peripheral")
        # print(thrpatpermaxlist)
        # print(thrpatpermax)
        # print("#distiguishing central from peripheral (perfectly identify peripheral based on max) #Above that central")
        # print(thrpatcenmaxlist)
        # print(thrpatcenmax)
        # print("#distiguishing central from peripheral (perfectly identify central based on min) #below that central")
        # print(thrpatcenminlist)
        # print(thrpatcenmin)
        # print("#distiguishing central from peripheral (perfectly identify peripheral based on min) #below that peripheral")
        # print(thrpatperminlist)
        # print(thrpatpermin)
        
        
        # normpatpat=[]
        # for y in range(len(thrnormhemaxlist)):
        #     normpatpat.append(thrnormhemaxlist[y]+str(thrnormhemax[y]))
        # for t in range(len(thrnormpatminlist)):
        #     normpatpat.append(thrnormpatminlist[t]+str(thrnormpatmin[t]))
        # print("#distiguishing healthy from patients (perfectly identify patients) ")
        # print(normpatpat)
        
        # normpatheal=[]
        # for y in range(len(thrnormpatmaxlist)):
        #     normpatheal.append(thrnormpatmaxlist[y]+str(thrnormpatmax[y]))
        # for t in range(len(thrnormheminlist)):
        #     normpatheal.append(thrnormheminlist[t]+str(thrnormhemin[t]))
        # print("#distiguishing healthy from patients (perfectly identify healthy) ")
        # print(normpatheal)
        
        # cenpercen=[]
        # for y in range(len(thrpatpermaxlist)):
        #     cenpercen.append(thrpatpermaxlist[y]+str(thrpatpermax[y]))
        # for t in range(len(thrpatperminlist)):
        #     cenpercen.append(thrpatperminlist[t]+str(thrpatpermin[t]))
        # print("#distiguishing central from peripheral (perfectly identify peripheral) ")
        # print(cenpercen)
        
        
        
   #We are out of the loop of images and inside p and l loop
   #and so, for each value of p and l, which indicate how many metrics are used to distinguish
   #peripheral from central palsy and healthy from patients respectively the accuracy, recall etc. are printed
   #along with graphs of how the accuracy improves by adding extra metrics in each case

        #Print the below only if we check healthy or if we are at the last patient metric that we check 
        if int(check_pat)==0 or p==pat1-1:
            print("For normal/pat the number of metric is")            
            print(l)
            
            print('\n')
            print('Below some numbers for normal/patients classification are presented:')
            print("TP")
            print(tpfin)
            print("FN")
            print(fnfin)
            print("TN")
            print(tnfin)
            print("FP")
            print(fpfin)
            print("FN list")
            print(fnfinlis)
            print("FP list")
            print(fpfinlis)
            print("Total True")
            print(tpfin+tnfin)
            print("Total False")
            print(fnfin+fpfin)
            print('Recall')
            re=tpfin/(tpfin+fnfin)
            print(re)
            print('Precision')
            if tpfin+fpfin==0:
                print("Not defined precision and F1 score")
                print('Accuracy')
                print("{:.2f}%".format(100*(tpfin+tnfin)/(tpfin+tnfin+fnfin+fpfin)))
            else:
                pr=tpfin/(tpfin+fpfin)
                print(pr)
                print('F1 Score')
                print(2*pr*re/(pr+re))
                print('Accuracy')
                print("{:.2f}%".format(100*(tpfin+tnfin)/(tpfin+tnfin+fnfin+fpfin)))
            
            if int(check_pat)==0:
                
                pointstoplot.append(100*(tpfin+tnfin)/(tpfin+tnfin+fnfin+fpfin))
                senspoints.append(re)
                if 100*(tpfin+tnfin)/(tpfin+tnfin+fnfin+fpfin)>initac and tpfin+tnfin>totaltrues:
                    initac=100*(tpfin+tnfin)/(tpfin+tnfin+fnfin+fpfin)
                    if int(manual_an)==1:
                        bestmetrics.append(metrnames[l]) 
                    else:
                        bestmetrics.append(metrnames2[l])
                    totaltrues=tpfin+tnfin

            
        
        
#Print the below only if we check petients or if we are at the last metric that we check for healthy
    if int(check_pat)==1 or l==heal1-1:
        #Patients
        print('\n')
        print("For cen/per the number of metric is")            
        print(p)
        print('\n')
        print('Below some numbers for peripheral/central classification are presented')
        print("TP")
        print(tpfin1)
        print("FN")
        print(fnfin1)
        print("TN")
        print(tnfin1)
        print("FP")
        print(fpfin1)
        print("FN list")
        print(fnfinlis1)
        print("FP list")
        print(fpfinlis1)
        print("Total True")
        print(tpfin1+tnfin1)
        print("Total False")
        print(fnfin1+fpfin1)
        print('Recall')
        re1=tpfin1/(tpfin1+fnfin1)
        print(re1)
        print('Precision')
        if tpfin1+fpfin1==0:
               print("Not defined precision and F1 score")
               print('Accuracy')
               print("{:.2f}%".format(100*(tpfin1+tnfin1)/(tpfin1+tnfin1+fnfin1+fpfin1)))
        else:
               pr1=tpfin1/(tpfin1+fpfin1)
               print(pr1)
               print('F1 Score')
               print(2*pr1*re1/(pr1+re1))
               print('Accuracy')
               print("{:.2f}%".format(100*(tpfin1+tnfin1)/(tpfin1+tnfin1+fnfin1+fpfin1)))
        print('Accuracy for peripheral')
        print("{:.2f}%".format(100*(tpfin1)/(tpfin1+fnfin1)))
        print('Accuracy for central')
        print("{:.2f}%".format(100*(tnfin1)/(tnfin1+fpfin1)))
        print('\n')
        print('Accuracy for a random image is')
        print("{:.2f}%".format((100*(tpfin1+tnfin1)/(tpfin1+tnfin1+fnfin1+fpfin1))*(tpfin+tnfin)/(tpfin+tnfin+fnfin+fpfin)))
        
        if int(check_pat)==1:
            pointstoplotpat.append(100*(tpfin1+tnfin1)/(tpfin1+tnfin1+fnfin1+fpfin1))
            senspointspat.append(re1)
            if 100*(tpfin1+tnfin1)/(tpfin1+tnfin1+fnfin1+fpfin1)>initac and tpfin1+tnfin1>totaltrues:
                initac=100*(tpfin1+tnfin1)/(tpfin1+tnfin1+fnfin1+fpfin1)
                if int(manual_an)==1:
                    bestmetrics.append(metrnamespat[p]) 
                else:
                    bestmetrics.append(metrnamespat2[p])
                totaltrues=tpfin1+tnfin1
    

print(bestmetrics)
print(initac)
print(totaltrues)


if int(manual_an)==1: 
    nums=[i for i in range(1,len(metrnames)+1)]
    length=len(metrnames)
else:
    nums=[i for i in range(1,len(metrnames2)+1)]
    length=len(metrnames2)

if length==len(pointstoplot) and int(manual_an)==1 and int(check_pat)==0:  
    plt.yticks(np.arange(min(pointstoplot), 100, 5))
    plt.plot(nums,pointstoplot,'-o')
    plt.xlabel("Number of threshold values for norm/pat")
    plt.ylabel("Accuracy for manual annotation")
    plt.show()
    #Activate below only to get a graph of recall
    # plt.yticks(np.arange(min(senspoints), 1, 0.05))
    # plt.plot(nums,senspoints,'-o')
    # plt.xlabel("Number of threshold values for norm/pat")
    # plt.ylabel("Sensitivity/Recall for manual annotation")
    # plt.show()
elif length==len(pointstoplot) and int(check_pat)==0:  
    plt.yticks(np.arange(min(pointstoplot), 100, 5))
    plt.plot(nums,pointstoplot,'-o')
    plt.xlabel("Number of threshold values for norm/pat")
    plt.ylabel("Accuracy for automatic annotation")
    plt.show()
    #Activate below only to get a graph of recall
    # plt.yticks(np.arange(min(senspoints), 1, 0.05))
    # plt.plot(nums,senspoints,'-o')
    # plt.xlabel("Number of threshold values for norm/pat")
    # plt.ylabel("Sensitivity/Recall for automatic annotation")
    # plt.show()


if int(manual_an)==1: 
    nums2=[i for i in range(1,len(metrnamespat)+1)]
    length2=len(metrnamespat)
else:
    nums2=[i for i in range(1,len(metrnamespat2)+1)]
    length2=len(metrnamespat2)
# #plt.scatter(nums,pointstoplot)
if length2==len(pointstoplotpat) and int(manual_an)==1 and int(check_pat)==1:
    plt.yticks(np.arange(min(pointstoplotpat), 100, 5))
    plt.plot(nums2,pointstoplotpat,'-o')
    plt.xlabel("Number of threshold values for per/cen")
    plt.ylabel("Accuracy for manual annotation")
    plt.show()
    #Activate below only to get a graph of recall
    # plt.yticks(np.arange(min(senspointspat), 1, 0.05))
    # plt.plot(nums2,senspointspat,'-o')
    # plt.xlabel("Number of threshold values for per/cen")
    # plt.ylabel("Sensitivity/Recall for manual annotation")
    # plt.show()
elif length2==len(pointstoplotpat) and int(check_pat)==1:
    plt.yticks(np.arange(min(pointstoplotpat), 100, 5))
    plt.plot(nums2,pointstoplotpat,'-o')
    plt.xlabel("Number of threshold values for per/cen")
    plt.ylabel("Accuracy for automatic annotation")
    plt.show()
    #Activate below only to get a graph of recall
    # plt.yticks(np.arange(min(senspointspat), 1, 0.05))
    # plt.plot(nums2,senspointspat,'-o')
    # plt.xlabel("Number of threshold values for per/cen")
    # plt.ylabel("Sensitivity/Recall for automatic annotation")
    # plt.show()

end = time.time()
print('\n')
print('Time in seconds')
print(end-start)
print('\n')
