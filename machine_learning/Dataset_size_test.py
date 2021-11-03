# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:40:01 2021

@author: hendr
"""
import numpy as np
import pandas as pd

df = np.load("C:/Users/hendr/Documents/TW studie jaar 3/palsy-master/machine_learning/features.npy", allow_pickle = True)
df = pd.DataFrame(df)
Peripheral_subset = df.iloc[:103, :]
Central_subset = df.iloc[103:143, :]
Healthy_subset = df.iloc[143:, :]
Peripheral_subset = Peripheral_subset.drop([102]) # leave out the one broken datapoint
patient_list = [Peripheral_subset, Central_subset, Healthy_subset]


#List with testing sizes
Size_testing_list = [50, 100, 150, 200]
Peripheral_scale = len(Peripheral_subset)/(len(df) - 1)
Central_scale = len(Central_subset)/(len(df) - 1)
Healthy_scale = len(Healthy_subset)/(len(df) - 1)

for size in Size_testing_list:
    Peripheral_subset_sized = pd.DataFrame(Peripheral_subset.sample(n = int(Peripheral_scale*size)))
    Central_subset_sized = pd.DataFrame(Central_subset.sample(n = int(Central_scale*size)))
    Healthy_subset_sized = pd.DataFrame(Healthy_subset.sample(n = int(Healthy_scale*size)))
    
    df_sized = pd.concat([Peripheral_subset_sized, Central_subset_sized, Healthy_subset_sized])
    if size == 50:
        print(df_sized)
        print(Peripheral_subset_sized)
        print(Central_subset_sized)
        
    #Algoritmes worden hier uitgevoerd
    
    #Resultaten worden geappend in een lijst 

#Lijndiagram wordt gemaakt met de gegevens uit de lijst
