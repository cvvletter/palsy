import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# set the random seed so that results are reproducible
np.random.seed(42)

# load all features and labels of the dataset
features, labels = np.load('./machine_learning/features.npy'), np.load('./machine_learning/labels.npy')

# remove the one broken data point
features, labels = np.delete(features, 102, 0), np.delete(labels, 102, 0)

# print the dataset summary
# print("Features:", features.shape, "\nLabels:  ", labels.shape)

# calculations for how many data points have to be removed each time.
count_classes = np.bincount(labels)
periphl_scale = count_classes[0]/len(features)
central_scale = count_classes[1]/len(features)
healthy_scale = count_classes[2]/len(features)
multiplier = 0.05*len(features)
end_size = 0.2*len(features)

accuracy_svm = []
# accuracy_naivebayes = []
dataset_size = []

while len(features) > end_size:
    # print("before",len(features))
    count_classes = np.bincount(labels)
    for i in range (int(round(periphl_scale*multiplier))): # remove some peripheral,
        features = np.delete(features, 0, 0)
        labels = np.delete(labels, 0, 0)
    for i in range (int(round(central_scale*multiplier))): # some central,
        features = np.delete(features, 0+count_classes[0], 0)
        labels = np.delete(labels, 0+count_classes[0], 0)
    for i in range (int(round(healthy_scale*multiplier))): # and some healthy datapoints
        features = np.delete(features, 0+count_classes[0]+count_classes[1], 0)
        labels = np.delete(labels, 0+count_classes[0]+count_classes[1], 0)
    # print("after",len(features))

    # calculate accuracy for svm
    correct = 0
    n = len(features)
    for i in range(len(labels)):
        x_test = features[i]
        x_test = np.reshape(x_test, ([1,-1]))
        y_test = labels[i]
        x_train_loocv = np.delete(features, i, 0)
        y_train_loocv = np.delete(labels, i, 0)
        modelsvm = svm.SVC(kernel='poly',degree=5,class_weight='balanced')
        modelsvm.fit(x_train_loocv, y_train_loocv)
        prediction = modelsvm.predict(x_test)
        if (prediction == y_test):
            correct += 1
    accuracy = correct/n
    # print("a=",accuracy)

    # store accuracy...
    accuracy_svm.append(accuracy)

    # ...and dataset size
    dataset_size.append(len(features))

# plot accuracy to dataset size
plt.plot(dataset_size, accuracy_svm)
plt.ylabel('Testing accuracy')
plt.xlabel('Dataset size')
plt.show()
