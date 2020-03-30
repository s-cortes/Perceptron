import numpy as np

class SimplePerceptron:

  def __init__(self, targets, training_set, weights=None, contains_bias=False,):
    if(len(targets.shape) > 1 or len(targets) != len(training_set)):
      return None

    self.weights = weights if weights is not None  else np.ones(training_set.shape[1])
    self.training_set = training_set
    self.targets = targets

    if(not contains_bias):
      np.insert(self.training_set, 0, 1.)
      np.c_[self.training_set, np.ones(self.training_set.shape[0])]

