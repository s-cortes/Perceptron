from .base_activation import ActivationFunction

class SigmoidActivation(ActivationFunction):
	'''Implementation of a Sigmoin Function'''

	def activate(self, input_value):
		return 1. if input_value >= 0 else 0.
	
	def update(self, t, x, w, output, lr, verbose=True):
		w_new = w + 2 * lr * (t - output) * x
		
		if verbose: print('\tw* = {} + {} * ({} - {}) * {} = {}'.format(
			w, lr, t, output, x, w_new))
		
		return w_new

	def __str__(self):
		return 'step'