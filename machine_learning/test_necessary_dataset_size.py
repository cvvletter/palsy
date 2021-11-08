import numpy as np
import matplotlib.pyplot as plt

# set the random seed so that results are reproducible
np.random.seed(42)

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
print(periphl_subset.shape, periphl_labels.shape)
print(central_subset.shape, central_labels.shape)
print(healthy_subset.shape, healthy_labels.shape)

periphl_scale = len(periphl_subset)/len(features)
central_scale = len(central_subset)/len(features)
healthy_scale = len(healthy_subset)/len(features)
multiplier = 0.05*len(features)
accuracy_array = []
size_array = []

while len(features) > 20:
    periphl_subset = periphl_subset[np.random.choice(len(periphl_subset), size = len(periphl_subset) - int(round(periphl_scale*multiplier)), replace=False), :]
    central_subset = central_subset[np.random.choice(len(central_subset), size = len(central_subset) - int(round(central_scale*multiplier)), replace=False), :]
    healthy_subset = healthy_subset[np.random.choice(len(healthy_subset), size = len(healthy_subset) - int(round(healthy_scale*multiplier)), replace=False), :]
    features = []
    features.append(periphl_subset)
    features.append(central_subset)
    features.append(healthy_subset)
    
    accuracy = len(features)
    
    accuracy_array.append(accuracy)
    size_array.append(len(features))
    print(len(features))
    
plt.plot(size_array, accuracy_array)
