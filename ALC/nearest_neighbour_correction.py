from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np
import copy

class NearestNeighbourCorrection:
    '''
    Abstract: Iterate through every sample and classifiy it using the k-nearest neighors.
              If the classifier predicts a different label with a confidence higher than some threshold,
              store the predicted label and mark the sample.
              After iterating through all samples once, all marked samples adopt their predicted label.
              Continue, till there is no more change in labels.


    Parameters:
    ----------

    K: {int}, default = 8
        Number of nearest neighbours to consider

    stopping_crit: {int} >= 0, default = 0

    conf_threshold: {float}, default = 0.5


    Methods:
    ----------

    .fit_transform(X, y)

    '''


    def __init__(self, K=8, stopping_crit = 0, conf_threshold = 0.5):
        self.K = K
        self.stopp = stopping_crit
        self.conf_threshold = conf_threshold
        self.knn = KNN(n_neighbors=self.K)


    def fit_transform(self, X, y):

        count = 0

        while True:
            count += 1

            y, n_correction = self._sweep(X, y)

            if n_correction <= self.stopp:
                break

            if count > 30:
                raise Exception('No stationary labels. Consider a higher stopping_crit.')

        return y


    def _sweep(self, X, y):

        y_corrected = copy.copy(y)
        n_correction = 0

        self.knn.fit(X, y)

        pred = self.knn.predict(X)
        pred_proba = self.knn.predict_proba(X)

        for i in range(len(X)):

            if np.max(pred_proba[i]) >= self.conf_threshold:

                if pred[i] != y[i]:
                    y_corrected[i] = pred[i]
                    n_correction += 1

        return y_corrected, n_correction
