from solver.perceptron import SimplePerceptron
import numpy as np

if __name__ == "__main__":
  x = np.array([[0,0,0], [0,2,1], [1,1,1], [1,-1,0]])
  t = np.array([-1,1,1,-1])

  perceptron = SimplePerceptron(t, x, bias=1)
  perceptron.train(2)
  print('Prediction', perceptron.predict(np.array([1,0,0,1])))