import numpy as np


def kmeans(data, k, thresh=1, max_iterations=100):
    centers =  data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(max_iterations):
        distance = np.linalg.norm(data[:, None] - centers, axis = 2)

        labels = np.argmin(distance, axis=1)
        new_centers = np.array([data[labels == i].mean(axis = 0) for i in range(k)])

        if np.all(centers == new_centers):
            break

        centers = new_centers
    
    return labels, centers