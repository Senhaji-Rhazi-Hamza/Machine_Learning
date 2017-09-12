import numpy as np
from mnist import load_mnist
from MajorityClassifier import MajorityClassifier
from RandomClassifier import RandomClassifier
from Lin1_Gauss import Lin1_Gauss
from Lin2_Gauss import Lin2_Gauss
from Lin3_Gauss import Lin3_Gauss

train_data, train_labels = load_mnist(dataset='training', path='../ressources')
test_data, test_labels = load_mnist(dataset='testing', path='../ressources')

train_data = np.reshape(train_data, (60000, 28 * 28)).T
test_data  = np.reshape(test_data,  (10000, 28 * 28)).T


#Random Classifer
Q1 = RandomClassifier()
Q1.train(train_data, train_labels)
perf = Q1.performance(test_data, test_labels)

#Majority Classifier
Q1 = MajorityClassifier()
Q1.train(train_data, train_labels)
perf = Q1.performance(test_data, test_labels)

#Lin1_Gauss Classifier
Q1 = Lin1_Gauss()
Q1.train(train_data, train_labels)
perf = Q1.performance(test_data, test_labels)

#Lin1_Gauss Classifier # a simple distance between the avg of each class and the image test
Q1 = Lin1_Gauss()
Q1.train(train_dataF, train_labels)
perf = Q1.performance(test_dataF, test_labels)
#Lin2_Gauss Classifier # mahanalobis distance  between the avg of each class and the image test considering all classes have same cov
Q1 = Lin2_Gauss()
Q1.train(train_data, train_labels)
perf = Q1.performance(test_data, test_labels)

#Lin3_Gauss Classifier 
# mahanalobis distance  between the avg of each class and the image test considering each classes have her own covariance
Q1 = Lin3_Gauss()
Q1.train(train_data, train_labels)
perf = Q1.performance(test_data, test_labels)
