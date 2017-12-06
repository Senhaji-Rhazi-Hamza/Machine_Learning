import numpy as np 
class Lin1_Gauss:
  def train(self, data_train, train_labels):  
    self.classes = np.unique(train_labels)
    self.classesAverage = np.array([None for i in range(len(self.classes))])
    self.set_vec_average(data_train, train_labels)
  def set_vec_average(self, data_train, train_labels):
    for i, label in enumerate(train_labels):
      if self.classesAverage[label] is None:
        self.classesAverage[label] = data_train[:,i]# [i,0] acess data, [i,1] acess data_tag
      else:
        self.classesAverage[label] = self.classesAverage[label] + data_train[:,i]
    count = np.bincount(train_labels) 
    for i in range(len(self.classesAverage)):
      self.classesAverage[i] = self.classesAverage[i] / count[self.classes[i]]
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
      distances[i] = np.linalg.norm(self.classesAverage[i] - vec_test)
    index_label = np.argmin(distances)
    return self.classes[index_label]
  def performance(self, test, labels_test ):
    ret = (self.process(test) == labels_test).sum() / labels_test.shape[0]
    print("the performance of LinearGaussclassifier is", ret) 
    return ret
