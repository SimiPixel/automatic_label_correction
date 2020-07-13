from sklearn.utils import shuffle
import numpy as np

def KFold(K, X, y_groud_truth, y_false):

    stepsize = X.shape[0] // K

    # Shuffle it
    X, y_groud_truth, y_false = shuffle(X, y_groud_truth, y_false)

    # Partioning
    train, test = [], []

    for fold in range(K):

        mask = np.arange(fold*stepsize, (fold+1)*stepsize)

        train.append((np.delete(X, mask, axis = 0), np.delete(y_false, mask, axis = 0)))

        test.append((X[mask], y_groud_truth[mask]))

    return train, test
