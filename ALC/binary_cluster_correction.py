## Multi-Class Unsupervised Classification with Label Correction of HRCT Lung Images
## Mithun Nagendra Prasad and Arcot Sowmya

# Idea: Take all samples out of X, with same class labels y. Use kMeans to find 2 cluster centroids, since all samples are either Correct or False.
#       Find main cluster centroid. Mark all samples that belong to main cluster.
#       Repeat for all classes. Train NNC using only marked samples.
#       Treat all unmarked samples as unlabeled. Use trained NNC to predict those labels. Treat as true labels. Done.
