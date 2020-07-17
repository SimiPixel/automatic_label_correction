import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier as KNN


class BinaryClusterCorrection:

    '''
    Abstract: Take all samples out of X, with same class labels y. Use kMeans to find 2 cluster centroids, since all samples are either Correct or False.
              Find main cluster centroid. Mark all samples that belong to main cluster.
              Repeat for all classes. Train KNN (Nearest neighbors classifier) using only marked samples.
              Treat all unmarked samples as unlabeled. Use trained KNN to predict those labels. Treat as true labels. Done.


    Methods:
    ----------

    .fit_transform(X, y):
        Returns: y_corrected


    References:
    ----------

        [1] Multi-Class Unsupervised Classification with Label Correction of HRCT Lung Images. Mithun Nagendra Prasad and Arcot Sowmya.

    '''

    def __init__(self):
        self.kmeans = KMeans(n_clusters=2)


    def fit_transform(self, X, y):

        N = X.shape[0]
        unique_labels = np.unique(y)

        # Marks mask
        marks = np.zeros(N)

        for label in unique_labels:

            mask_y = np.where(y==label, True, False)

            candidates = X[mask_y]

            # Fit kMeans and predict cluster membership
            membership = self.kmeans.fit_predict(candidates)

            # Get main cluster
            _1, _2 = np.unique(membership, return_counts=True)
            main_cluster = _1[np.argmax(_2)]

            mask_belongs_to_main = np.where(membership == main_cluster, True, False)

            # Mark samples that belong to main cluster
            count = -1
            for i in range(N):
                if mask_y[i] == True:
                    count += 1
                    if mask_belongs_to_main[count] == True:

                        # If is main cluster mark it
                        marks[i] = 1

        marks = marks.astype(bool)

        knn = KNN()
        knn.fit(X[marks], y[marks])

        new_labels = knn.predict(X[~marks])

        y[~marks] = new_labels

        return y
