import numpy as np 
class RandomClassifier:
  def train(self, train, label):
    self.classes = np.unique(label)
  def process(self, test):
    out = np.zeros(test.shape[1])
    for i in range(0, test.shape[1]):
      out[i] = np.random.choice(self.classes, 1)
    return out
  def performance(self, test, labels_test ):
    ret = (self.process(test) == labels_test).sum() / labels_test.shape[0]
    print("the performance of Randomclassifier is", ret) 
    return ret
