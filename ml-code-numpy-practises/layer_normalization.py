import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	# Your code here
	X_mean = np.mean(X, axis = -1, keepdims=True)
	X_variance = np.average(np.square(X-X_mean), axis = -1, keepdims=True)
	X_norm = (X - X_mean) / np.sqrt(X_variance + epsilon)
	scale = gamma * X_norm + beta
	return scale