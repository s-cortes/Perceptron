import abc

class ActivationFunction(abc.ABC):

	@abc.abstractmethod
	def activate(self, input_value):
		'''
		Base for implementing an activation function.
		@params:
			- input_value: Value to be used for cimputing the activation function.
		
		@returns: The result of the activation function (depends on the implementation)
		'''
		pass

	@abc.abstractmethod
	def update(self, t, x, w, output, lr, verbose=True):
		'''
		Function for updating the weights of the perceptron given the sample data and the target value, 
		given a particular activation function.

		@params:
			- t: Target value for the corresponding sample vector
			- x: Sample vector data.
			- w: weight vector.
			- output: output from the activation function
			- lr: Learning rate
			- verbose: Indicates if the function should execute the prints
		'''
		pass