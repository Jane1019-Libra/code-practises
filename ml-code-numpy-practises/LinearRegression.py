import numpy as np


def compute_cost(X, y, theta):
    m = len(y)
    prediction = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(y - prediction))
    return cost


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)

    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        prediction = X.dot(theta)
        error = prediction - y
        gradient = (1/m) * X.T.dot(error)

        theta = theta - alpha * gradient
        J_history[i] = compute_cost(X,y, theta)
    return theta, J_history