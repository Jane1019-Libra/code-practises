import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))


class LogisticRregression:

    def __init__(self, learning_rate = 0.01, num_iters=1000):
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        num_samples, num_features = X.shape

        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iters):
            linear_model = X.dot(self.weights) + self.bias
            y_pred = sigmoid(linear_model)
            dw = (1/num_samples) * X.T.dot(y_pred-y)
            dt = (1/num_samples) * np.sum(y_pred-y)

            self.weights -= dw * self.learning_rate
            self.bias -= dt * self.learning_rate
    
    def prediction_prob(self, X):
        return sigmoid(X.dot(self.weight) + self.bias)
    
    def predict(self, X, threshold = 0.5):
        y_pred_prob = self.prediction_prob(X)

        y_pred_cls = [1 if i > threshold else 0 for i in y_pred_prob]
        return y_pred_cls

