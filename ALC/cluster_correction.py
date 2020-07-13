import numpy as np
from sklearn.cluster import KMeans

class ClusterCorrection:

    '''
    Parameters:
    -----------

    number_of_clusterings: {int}, default = 10


    Methods:
    -----------

    .fit_transform(X, y):


    References:
    -----------

        [1] Label Noise Correction Methods. Bryce Nicholson, Victor S. Sheng, Jing Zhang, Zhiheng Wang.
    '''

    def __init__(self, number_of_clusterings = 10):

        self.number_of_clusterings = number_of_clusterings


    def fit_transform(self, X, y):

        ## set up characteristic quantities of data for later use
        ## nameing scheme in analogy to paper

        self.N = X.shape[0]
        self.n_features = X.shape[1]
        self.a = self.number_of_clusterings
        _1, _2 = np.unique(y, return_counts = True)
        self.n_unique_labels = len(_1)
        self.label_totals = _2/(np.sum(_2))

        # Initialize weight vector for later use
        ins_weights = np.zeros((self.N, self.n_unique_labels))

        for i in range(1, self.a+1):

            k = i/self.a * self.N/2 + 2
            k = int(k)

            ## Clustering
            kmeans = KMeans(n_clusters = k)
            cluster_indices = kmeans.fit_predict(X)

            ## set up characteristic quantities for the clusters

            # Label distribution of the k-th cluster
            cluster_label_distro = []
            for cluster_idx in range(k):
                mask = np.where(cluster_indices == cluster_idx, 1, 0).astype(bool)
                cluster_label_distro.append(np.unique(np.hstack((_1, y[mask])), return_counts = True)[1] - 1)
            cluster_label_distro = np.array(cluster_label_distro)
            # Convert counts to distribution
            cluster_label_distro = cluster_label_distro/np.sum(cluster_label_distro)

            # Number of samples that belong to each cluster
            size_of_cluster = np.unique(cluster_indices, return_counts = True)[1]

            ## CalcWeights
            calc_weights = []
            for cluster_idx in range(k):

                u = 1/self.n_unique_labels
                multiplier = min(np.log10(size_of_cluster[cluster_idx]), 2)
                calc_weights.append(multiplier * (cluster_label_distro[cluster_idx]-u)/self.label_totals)
            calc_weights = np.array(calc_weights)

            ## Add up class label weights for every instance for different i
            for j in range(self.N):
                ins_weights[j] += calc_weights[cluster_indices[j]]

        label_guess = np.argmax(ins_weights, axis = 1)
        return label_guess
