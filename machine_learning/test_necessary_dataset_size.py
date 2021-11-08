import numpy as np
import random
import os

# set the random seed so that results are reproducible
random.seed(42)

# load all features and labels of the dataset
features, labels = np.load('./machine_learning/features.npy'), np.load('./machine_learning/labels.npy')

# remove the one broken data point
features, labels = np.delete(features, 102, 0), np.delete(labels, 102, 0)

# print the dataset summary
print("Features:", features.shape, "\nLabels:  ", labels.shape)

# take subsets so the three diagnoses are separated
periphl_subset, periphl_labels = features[0:102,:], labels[0:102]
central_subset, central_labels = features[102:142,:], labels[102:142]
healthy_subset, healthy_labels = features[142:202,:], labels[142:202]

# print the subsets summaries to verify the subset splitting
# print(periphl_subset.shape, periphl_labels.shape)
# print(central_subset.shape, central_labels.shape)
# print(healthy_subset.shape, healthy_labels.shape)

periphl_scale = len(periphl_subset)/len(features)
central_scale = len(central_subset)/len(features)
healthy_scale = len(healthy_subset)/len(features)
multiplier = 0.05*len(features)
accuracy_array = []
size_array = []

while len(features) > 20:
    periphl_subset = random.sample(periphl_subset, len(periphl_subset) - int(round(periphl_scale*multiplier)))
    central_subset = random.sample(central_subset, len(central_subset) - int(round(central_scale*multiplier)))
    healthy_subset = random.sample(healthy_subset, len(healthy_subset) - int(round(healthy_scale*multiplier)))
    features = []
    features.append(periphl_subset)
    features.append(central_subset)
    features.append(healthy_subset)
    
    #bereken accuracy
    
    accuracy_array.append()
    size_array.append(len(features))
    
plot(accuracy_array, size_array)
