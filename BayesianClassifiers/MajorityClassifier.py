import numpy as np 
class MajorityClassifier:
  def train(self, train, label):
    self.MajorityClasse = np.bincount(label).argmax()
  def process(self, test):
    out = np.zeros(test.shape[1])
    for i in range(0, test.shape[1]):
      out[i] = self.MajorityClasse
    return out
  def performance(self, test, labels ):
    ret = (self.process(test) == labels).sum() / labels.shape[0]
    print("the performance of MajorityClassifier is", ret)
    return ret
