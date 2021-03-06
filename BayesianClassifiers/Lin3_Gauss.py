import numpy as np 
class Lin3_Gauss:
  def train(self, data_train, train_labels):  
    self.classes = np.unique(train_labels)
    self.count = np.bincount(train_labels) 
    self.classesAverage = np.array([None for i in range(len(self.classes))])
    self.set_vec_average(data_train, train_labels)
    self.covinverse = np.array([None for i in range(len(self.classes))])

    for i in range(len(self.classes)):
      self.covinverse[i] = np.linalg.pinv(np.cov(data_train[:,np.where(train_labels == self.classes[i])[0]])) 
  def set_vec_average(self, data_train, train_labels):
    for i, label in enumerate(train_labels):
      if self.classesAverage[label] is None:
        self.classesAverage[label] = data_train[:,i]# [i,0] acess data, [i,1] acess data_tag
      else:
        self.classesAverage[label] = self.classesAverage[label] + data_train[:,i]
    for i in range(len(self.classesAverage)):
      self.classesAverage[i] = self.classesAverage[i] / self.count[self.classes[i]]
    return self.classesAverage  

 #given a test_vector(an image) return the closest label to that image
  def process(self, test):
    out = np.zeros(test.shape[1])
    for i in range(0, test.shape[1]):
      out[i] = self.answer_out(test[:,i])
    return out
  def answer_out(self, vec_test):
    distances = [None for i in range(len(self.classesAverage))]
    for i in range(len(self.classesAverage)):
      distances[i] = np.dot(np.dot(np.transpose(self.classesAverage[i]- vec_test), self.covinverse[i]), (self.classesAverage[i] - vec_test))
    index_label = np.argmin(distances)
    return self.classes[index_label]
  def performance(self, test, labels_test ):
    ret = (self.process(test) == labels_test).sum() / labels_test.shape[0]
    print("the performance of Linear3Gaussclassifier is", ret) 
    return ret
