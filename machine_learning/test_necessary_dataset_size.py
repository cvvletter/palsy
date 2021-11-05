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

periphl_scale, central_scale, healthy_scale = 5, 2, 3
for i in range (10,20+1):
    features_test, labels_test = [], []

    # append the random peripheral data points
    features_test = features_test.append(random.sample(periphl_subset, int(periphl_scale*i)))
    labels_test = labels_test.append(random.sample(periphl_labels, int(periphl_scale*i)))

    # append the random central data points
    features_test = features_test.append(random.sample(central_subset, int(central_scale*i)))
    labels_test = labels_test.append(random.sample(central_labels, int(central_scale*i)))

    # append the random healthy data points
    features_test = features_test.append(random.sample(healthy_subset, int(healthy_scale*i)))
    labels_test = labels_test.append(random.sample(healthy_labels, int(healthy_scale*i)))