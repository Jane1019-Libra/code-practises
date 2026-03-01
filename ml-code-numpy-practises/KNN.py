import numpy as np
from collections import Counter



class KNN:
    def __init__(self, k=3):
        self.k = k

    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []

        for x in X_test:
            distance = np.sqrt(np.sum((x - self.X_train)**2, axis = 1))

            nearest_distance = np.argsort(distance)[:self.k]
            nearest_label = self.y_train[nearest_distance]
            most_common = Counter(nearest_label).most_common(1)[0][0]

            predictions.append(most_common)
        return np.array(predictions)
