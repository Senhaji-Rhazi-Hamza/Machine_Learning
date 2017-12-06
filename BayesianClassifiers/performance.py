import numpy as np
from mnist import load_mnist
from MajorityClassifier import MajorityClassifier
from RandomClassifier import RandomClassifier
from Lin1_Gauss import Lin1_Gauss
from Lin2_Gauss import Lin2_Gauss
from Lin3_Gauss import Lin3_Gauss
import sys
import time
sys.path.append('../PCA_LDA')
import projectors as p


train_data, train_labels = load_mnist(dataset='training', path='../ressources')
test_data, test_labels = load_mnist(dataset='testing', path='../ressources')

train_data = np.reshape(train_data, (60000, 28 * 28)).T
test_data  = np.reshape(test_data,  (10000, 28 * 28)).T

#Reduction of dimensionality using PCA
V = p.extractPCAEIGV(train_data, 0.75)
train_dataF =(train_data.T.dot(V)).T
test_dataF = (test_data.T.dot(V)).T
V = p.extractLDA(train_data, train_labels, 15)
train_dataG =(train_data.T.dot(V)).T
test_dataG = (test_data.T.dot(V)).T

t0 = time.time()
#Random Classifer
Q1 = RandomClassifier()
Q1.train(train_data, train_labels)
perf = Q1.performance(test_data, test_labels)
print("time exec", time.time() - t0)

t0 = time.time()
#Majority Classifier
Q1 = MajorityClassifier()
Q1.train(train_data, train_labels)
perf = Q1.performance(test_data, test_labels)
print("time exec", time.time() - t0)


t0 = time.time()
#Lin1_Gauss Classifier # a simple distance between the avg of each class and the image test
Q1 = Lin1_Gauss()
Q1.train(train_data, train_labels)
perf = Q1.performance(test_data, test_labels)
print("time exec", time.time() - t0)

t0 = time.time()
#Lin2_Gauss Classifier # mahanalobis distance  between the avg of each class and the image test considering all classes have same cov
Q1 = Lin2_Gauss()
Q1.train(train_data, train_labels)
perf = Q1.performance(test_data, test_labels)
print("time exec", time.time() - t0)

t0 = time.time()
#Lin3_Gauss Classifier 
# mahanalobis distance  between the avg of each class and the image test considering each classes have her own covariance
Q1 = Lin3_Gauss()
Q1.train(train_data, train_labels)
perf = Q1.performance(test_data, test_labels)
print("time exec", time.time() - t0)


print("Reduction PCA")
t0 = time.time()
#Lin3_Gaus Classifier with reduction dimentionality 
Q1 = Lin3_Gauss()
Q1.train(train_dataF, train_labels)
perf = Q1.performance(test_dataF, test_labels)
print("time exec", time.time() - t0)

print("Reduction LDA")
t0 = time.time()
#Lin3_Gaus Classifier with reduction dimentionality 
Q1 = Lin3_Gauss()
Q1.train(train_dataG, train_labels)
perf = Q1.performance(test_dataG, test_labels)
print("time exec", time.time() - t0)
