import numpy as np
from .activations.sigmoid_activation import SigmoidActivation

class SimplePerceptron:

	def __init__(self, targets, training_set, activation=SigmoidActivation, weights=None, bias=None, learning_rate=1):
		'''
		Constructor of the class. Initilizes the elements needed for training a perceptron.

		@params:
			- targets: Output vector consisting of D desired outputs.
			- training_set: Input verctor consisting of D inputs of dimension N.
			- activation: Activation function of the perceptron
			- weights: Weight vector consisting of D weights.
			- bias: Value of the bias. if it's None, the constructure that the trainig set and weight already contain it
		'''
		if len(targets.shape) > 1:
			raise ValueError('The targets should be a 1-dimensional array')
		elif len(targets) != len(training_set):
			raise ValueError('The number of targets ({}) differs from the number of training samples ({})'
				.format(len(targets), len(training_set)))
		elif weights is not None and len(weights) != len(training_set):
			raise ValueError('The number weights ({}) differs from the number of training samples ({})'
				.format(len(weights), len(training_set)))

		self.targets = targets
		self.activation = activation( )
		self.training_set = training_set
		self.weights = np.ones(training_set.shape[1]) if weights is None else weights
		self.learning_rate = learning_rate
		
		if bias is not None:
			self.weights = np.insert(self.weights, 0, 1. if bias is None else float(bias))
			self.training_set = np.c_[np.ones(self.training_set.shape[0]), self.training_set]
	
	def predict(self, query):
		if len(self.weights) != len(query):
			raise ValueError('The number of weights ({}) differs from the number of the query vector ({})'
				.format(len(self.weights), len(query)))
		return self.activation.activate(np.dot(query, self.weights)) 

	def train(self, max_epochs, verbose=True):
		'''
		Function for training the perceptron until it reaches convergences, or reaches a maximum number
		of epochs.

		@params:
			- max_epochs: Maximum number of epochs to execute during the training
			- verbose: Indicates if the function should execute the prints
		'''
		converged, input_value, output = True, 0, 0
		for epoch in range(max_epochs):
			print('-'*20, 'Epoch {}'.format(epoch + 1), '-'*20)
			converged = True
			for i, x in enumerate(self.training_set):
				input_value = np.dot(x, self.weights)
				output = self.activation.activate(input_value)
				
				if verbose:
					print('\tf(xw) = {}({} * {}) = {}({}) = {}'.format(
						self.activation, x, self.weights, self.activation, input_value, output))
				
				converged = converged and (self.targets[i] == output)
				if not converged: 
					self.activation.update(self.targets[i], x, self.weights, output, self.learning_rate, verbose)
			
			if converged: break
		if verbose:
			print('-'*20, ' END ', '-'*20)
			print('Training {} reached convergence'.format('DID' if converged else 'DID NOT'))

	def __str__(self):
		'''
		Custom representation of the class and its attributes.
		'''
		return 'Training Set {}:\n{}\n\nWeights{}:\n{}\n\n'.format(
			self.training_set.shape, self.training_set, self.weights.shape, self.weights)
