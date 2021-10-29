# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:19:34 2021

@author: hendr
"""
import numpy as np
import pandas as pd
import math as m
from sklearn.linear_model import LinearRegression


df = np.load("C:/Users/hendr/Documents/TW studie jaar 3/palsy-master/machine_learning/features.npy", allow_pickle = True)
df = pd.DataFrame(df)
Peripheral_subset = df.iloc[:103, :]
Central_subset = df.iloc[103:143, :]
Healthy_subset = df.iloc[143:, :]
Peripheral_subset = Peripheral_subset.drop([102]) # leave out the one broken datapoint
patient_list = [Peripheral_subset, Central_subset, Healthy_subset]
Measure_columns = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", 
                                       "M11", "M12", "M13", "M14", "M15", "M16", "M17", "M18", "M19", 
                                       "M20", "M21", "M22", "M23", "M24", "M25", "M26", "M27", "M28", 
                                       "M29", "M30", "M31", "M32", "M33", "M34", "M35", "M36", "M37", 
                                       "M38", "M39", "M40", "M41", "M42", "M43", "M44", "M45", "M46", 
                                       "M47", "M48", "M49", "M50", "M51", "M52", "M53"]
Metric_dataset = pd.DataFrame(columns = Measure_columns)
patient_counter = 0

#Calculates the area of a triangle with heron's formula 
def Area_triangle(point1,point2,point3):
    return 1/2*( point1[0]*(point2[1] -point3[1]) + point2[0]*(point3[1] - point1[1]) + point3[0]*(point1[1] - point2[1]) )

#Calculates the euclidian distance with the pythagorean theorem 
def Distance_calculator(point1, point2):
    return abs( m.sqrt((point1[0] - point2[0])**2) + m.sqrt((point1[1] - point2[1])**2) )

#Takes the slope of a linear regression algorithm based on the list of points given
def Best_slope(list_of_points):
    list_of_points = np.array(list_of_points)
    X = list_of_points[:,0].reshape(-1,1)
    y = list_of_points[:,1]
    model = LinearRegression()
    model.fit(X,y)
    return model.coef_
  
#Makes a ratio by dividing the larger side by the smaller side  
def Ratio_calculator(left, right):
    if left > right:
        return left/right
    elif right >= left:
        return right/left

#Loops the metrics for each patient in the dataset in the patient_list    
for dataset in patient_list:
    for patient in np.array(dataset):
        
        x_list = patient[0::2]
        y_list = patient[1::2]
        patient = np.array((x_list, y_list))
        patient = np.transpose(patient)
        
        #Defines general variables used in the metrics
        mean_eyelid_l = sum(patient[36:42])/len(patient[36:42])
        mean_eyelid_r = sum(patient[42:48])/len(patient[42:48])
        corner_mouth_l = patient[48]
        corner_mouth_r = patient[54]
        corner_nose_l = patient[31]
        corner_nose_r = patient[35]
        tip_nose = patient[30]
        upper_left_mouth = sum(patient[48:51])/len(patient[48:51])
        lower_right_mouth = sum(patient[54:57])/len(patient[54:57])
        #Min and max is switched because of the reversed y-axis from the landmark detection system
        lowest_chin = [i for i in patient if i[1] == max(patient[6:11][:,1])]
        lowest_chin = sum(lowest_chin)/len(lowest_chin)
        max_eyelid_l = [i for i in patient if i[1] == min(patient[37:39][:,1])][0]
        min_eyelid_l = [i for i in patient if i[1] == max(patient[40:42][:,1])][0]
        max_eyelid_r = [i for i in patient if i[1] == min(patient[:,1][43:45])][0]
        min_eyelid_r = [i for i in patient if i[1] == max(patient[:,1][46:48])][0]
        width_eye_l = abs(patient[39][0] - patient[36][0])
        width_eye_r = abs(patient[42][0] - patient[45][0])
        max_brow_l = [i for i in patient if i[1] == min(patient[17:22][:,1])][0]
        max_brow_r = [i for i in patient if i[1] == min(patient[22:27][:,1])][0]
        min_brow_l = [i for i in patient if i[1] == max(patient[17:22][:,1])][0]
        min_brow_r = [i for i in patient if i[1] == max(patient[22:27][:,1])][0]
        mean_mouth_corner_l = (sum(patient[48:50]) + patient[59])/3
        mean_mouth_corner_r = sum(patient[53:56])/len(patient[53:56])
        lowest_mouth_corner_r = [i for i in patient if i[1] == max(patient[53:56][:,1])][0]
        highest_mouth_corner_r = [i for i in patient if i[1] == min(patient[53:56][:,1])][0]
        highest_mouth_corner_l = [i for i in patient if i[1] == min(patient[48][1],patient[49][1],patient[59][1])][0]
        lowest_mouth_corner_l = [i for i in patient if i[1] == max(patient[48][1],patient[49][1],patient[59][1])][0]
        mean_lowerleft_eye_l = (sum(patient[40:42]) + patient[36])/3
        mean_upperright_eye_l = sum(patient[37:40])/len(patient[37:40])
        mean_lowerleft_eye_r = (sum(patient[46:48] + patient[42]))/3
        mean_upperright_eye_r = sum(patient[43:46])/len(patient[43:46])
        
        #For every distance calculated, the height difference is measured using the euclidian diagonal distance instead
        #With exceptions to measurements where switching this could significantly alter the result
        #A ratio is calculated for each metric that compared the left side of the face to the right
        #Each of these metrics are based on the measurements done by Nikos Sourlos 
        #Sourlos, N. (2020). Facial Imaging and Diagnosis System for Neurological Disorders (Master’s thesis, Delft University of Technology, Delft, The Netherlands). http://resolver.tudelft.nl/uuid:e60a60fc-073c-48b4-83af-d5823f309539
        
        #Metric 1 
        M1_l = Distance_calculator(mean_eyelid_l, corner_mouth_l)
        M1_r = Distance_calculator(mean_eyelid_r, corner_mouth_r)
        M1 = Ratio_calculator(M1_l, M1_r)
        
        #Metric 2
        M2_l = Distance_calculator(mean_eyelid_l, corner_nose_l)
        M2_r = Distance_calculator(mean_eyelid_r, corner_nose_r)
        M2 = Ratio_calculator(M2_l, M2_r)
        
        #Metric 3
        M3_l = Distance_calculator(mean_eyelid_l, tip_nose)
        M3_r = Distance_calculator(mean_eyelid_r, tip_nose)
        M3 = Ratio_calculator(M3_l, M3_r)
        
        #Metric 4 (points 49 to 51 are used instead of 51 to 53 to indicate the real upper left corner of the mouth)
        M4_l = Distance_calculator(mean_eyelid_l, upper_left_mouth)
        M4_r = Distance_calculator(mean_eyelid_r, upper_left_mouth)
        M4 = Ratio_calculator(M4_l, M4_r)
        
        #Metric 5
        M5_l = Distance_calculator(mean_eyelid_l, lower_right_mouth)
        M5_r = Distance_calculator(mean_eyelid_r, lower_right_mouth)
        M5 = Ratio_calculator(M5_l, M5_r)
        
        #Metric 6
        M6_l = Distance_calculator(tip_nose, corner_mouth_l)
        M6_r = Distance_calculator(tip_nose, corner_mouth_r)
        M6 = Ratio_calculator(M6_l, M6_r)
        
        #Metric 7 (isn't concrete if there are multiple minima, because of the geometric translation of the chin datapoints)
        M7_l = Distance_calculator(mean_eyelid_l, lowest_chin)
        M7_r = Distance_calculator(mean_eyelid_r, lowest_chin)
        M7 = Ratio_calculator(M7_l, M7_r)
        
        #Metric 8
        M8_l = abs( max_eyelid_l[1] - min_eyelid_l[1])
        M8_r = abs( max_eyelid_r[1] - min_eyelid_r[1])
        M8 = Ratio_calculator(M8_l, M8_r)
        
        #Metric 9  (ratio eye width/height)
        M9_l = width_eye_l/abs(max_eyelid_l[1] - min_eyelid_l[1])
        M9_r = width_eye_r/abs(max_eyelid_r[1] - min_eyelid_r[1])
        M9 = Ratio_calculator(M9_l, M9_r)
        
        #Metric 10 (surface area eye)
        M10_l = (Area_triangle(patient[36],patient[37],patient[41]) + 
                 Area_triangle(patient[37],patient[40],patient[41]) +
                 Area_triangle(patient[37],patient[38],patient[40]) +
                 Area_triangle(patient[38],patient[39],patient[40]))
        M10_r = (Area_triangle(patient[42],patient[43],patient[47]) + 
                 Area_triangle(patient[43],patient[44],patient[47]) +
                 Area_triangle(patient[44],patient[46],patient[47]) +
                 Area_triangle(patient[44],patient[45],patient[46]))
        M10 = Ratio_calculator(M10_l, M10_r)
        
        #Metric 11
        M11_l = width_eye_l
        M11_r = width_eye_r
        M11 = Ratio_calculator(M11_l, M11_r)
        
        #Metric 12
        M12_l = abs(max_brow_l[1] - min_eyelid_l[1])
        M12_r = abs(max_brow_r[1] - min_eyelid_r[1])
        M12 = Ratio_calculator(M12_l, M12_r)
        
        #Metric 13
        M13_l = Distance_calculator(tip_nose, max_brow_l)
        M13_r = Distance_calculator(tip_nose, max_brow_r)
        M13 = Ratio_calculator(M13_l, M13_r)
        
        #Metric 14
        M14_l = Distance_calculator(corner_nose_l, max_brow_l)
        M14_r = Distance_calculator(corner_nose_r, max_brow_r)
        M14 = Ratio_calculator(M14_l, M14_r)
        
        #Metric 15
        M15_l = Distance_calculator(max_brow_l, mean_mouth_corner_l)
        M15_r = Distance_calculator(max_brow_r, mean_mouth_corner_r)
        M15 = Ratio_calculator(M15_l, M15_r)
        
        #Metric 16
        M16_l = Distance_calculator(lowest_mouth_corner_l, max_eyelid_l)
        M16_r = Distance_calculator(highest_mouth_corner_r, max_eyelid_r)
        M16 = Ratio_calculator(M16_l, M16_r)
        
        #Metric 17
        M17_l = Distance_calculator(highest_mouth_corner_l, max_eyelid_l)
        M17_r = Distance_calculator(lowest_mouth_corner_r, max_eyelid_r)
        M17 = Ratio_calculator(M17_l, M17_r)
        
        #Metric 18
        M18 = abs(tip_nose[1] - lowest_chin[1])
        
        #Metric 19
        M19_l = Distance_calculator(mean_lowerleft_eye_l, mean_upperright_eye_l)
        M19_r = Distance_calculator(mean_lowerleft_eye_r, mean_upperright_eye_r)
        M19 = Ratio_calculator(M19_l, M19_r)
        
        #Metric 20
        M20_l = highest_mouth_corner_l[1]
        M20_r = highest_mouth_corner_r[1]
        M20 = Ratio_calculator(M20_l, M20_r)
        
        #Metric 21
        M21 = float(Best_slope(patient[17:22]))
        
        #Metric 22
        M22 = float(Best_slope(patient[22:27]))
        
        #Metric 23
        M23 = float(abs(M21 - M22))
        
        #Metric 24
        M24_l = Distance_calculator(max_eyelid_l, max_brow_l)
        M24_r = Distance_calculator(max_eyelid_r, max_brow_r)
        M24 = float(Ratio_calculator(M24_l, M24_r))
        
        #Metric 25
        M25 = float(Best_slope(patient[17:20]))
        
        #Metric 26
        M26 = float(Best_slope(patient[22:25]))
        
        #Metric 27 (Used points 19 till 21 instead)
        M27 = float(Best_slope(patient[18:21]))
        
        #Metric 28 (Used points 24 till 26 instead)
        M28 = float(Best_slope(patient[23:26]))
        
        #Metric 29 (Used points 37,38,42 instead)
        M29 = float(Best_slope([patient[36], patient[37] ,patient[41]]))
        
        #Metric 30 (Used points 43,44,48 instead)
        M30 = float(Best_slope([patient[42], patient[43] ,patient[47]]))
        
        #Metric 31 (points 37,38,39,42 were chosen for the left eye and 44 till 47 for the right eye)
        M31_l = max(Distance_calculator(patient[33], patient[36]), Distance_calculator(patient[33], patient[37]),
                    Distance_calculator(patient[33], patient[38]), Distance_calculator(patient[33], patient[41]))
        M31_r = max(Distance_calculator(patient[33], patient[43]), Distance_calculator(patient[33], patient[44]),
                    Distance_calculator(patient[33], patient[45]), Distance_calculator(patient[33], patient[46]))
        M31 = Ratio_calculator(M31_l, M31_r)
        
        #Metric 32
        M32 = float(Best_slope(patient[36:38]))
        
        #Metric 33
        M33 = float(Best_slope(patient[42:44]))
        
        #Metric 34
        M34 = float(abs(M32 - M33))
        
        #Metric 35 
        M35 = float(Best_slope(patient[38:40]))
        
        #Metric 36
        M36 = float(Best_slope(patient[44:46]))
        
        #Metric 37
        M37 = float(Best_slope(patient[17:19]))
        
        #Metric 38
        M38 = float(Best_slope(patient[22:24]))
        
        #Metric 39
        M39 = float(Best_slope(patient[20:22]))
        
        #Metric 40
        M40 = float(Best_slope(patient[25:27]))
        
        #Metric 41
        M41 = float(Best_slope([patient[36], patient[41]]))
        
        #Metric 42
        M42 = float(Best_slope([patient[42], patient[47]]))
        
        #Metric 43
        M43 = float(Best_slope(patient[39:41]))
        
        #Metric 44
        M44 = float(Best_slope(patient[45:47]))
        
        #Metric 45
        M45 = abs(max(max_eyelid_l[1], max_eyelid_r[1]) - min(min_eyelid_l[1], min_eyelid_r[1]))
        
        #Metric 46
        M46 = abs(max(max_brow_l[1], max_brow_r[1]) - min(min_brow_l[1], min_brow_r[1]))
        
        #Metric 47
        M47 = abs(max(max_brow_l[1], max_brow_r[1]) - min(min_eyelid_l[1], min_eyelid_r[1]))

        #Metric 48
        M48 = abs(max(max_brow_l[1], max_brow_r[1]) - max(max_eyelid_l[1], max_eyelid_r[1]))
        
        #Metric 49
        M49 = abs(max(max_brow_l[1], max_brow_r[1]) - min(lowest_mouth_corner_l[1], lowest_mouth_corner_r[1], patient[60][1], patient[64][1]))
        
        #Metric 50
        M50 = abs(abs(max_eyelid_l[1] - min_eyelid_l[1]) - abs(max_eyelid_r[1] - min_eyelid_r[1]))
        
        #Metric 51
        M51 = abs(max_brow_l[1] - max_brow_r[1])
        
        #Metric 52
        M52 = min_eyelid_r[1]
        
        #Metric 53
        M53 = min_eyelid_l[1]
        
        New_row = pd.DataFrame([[M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, 
                                M16, M17, M18, M19, M20, M21, M22, M23, M24, M25, M26, M27, M28, 
                                M29, M30, M31, M32, M33, M34, M35, M36, M37, M38, M39, M40, M41, 
                                M42, M43, M44, M45, M46, M47, M48, M49, M50, M51, M52, M53]], columns = Measure_columns)
        Metric_dataset = Metric_dataset.append(New_row, ignore_index = True) 
        patient_counter += 1
        

np.save("Metric_dataset",Metric_dataset)