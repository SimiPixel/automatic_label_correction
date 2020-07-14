import copy
import numpy as np

def Falsify(y, p, random_state = None):
    '''
    y: {np.array} of shape (n_samples)
    p: {float} between 0 and 1.
        Percentage of false labels afterwards

    Returns: false labels

    '''
    np.random.seed(random_state)

    y = copy.copy(y)

    unique_l, class_ratio = np.unique(y, return_counts=True)
    class_ratio = class_ratio / len(y)

    def reassign(l):
        temp_class_ratio = copy.copy(class_ratio)
        temp_class_ratio[l] = 0
        temp_class_ratio = temp_class_ratio / temp_class_ratio.sum()
        return np.random.choice(len(unique_l), 1, p = temp_class_ratio)[0]

    for idx in np.random.choice(len(y), int(p*len(y)), replace = False).astype(int):
        y[idx] = reassign(y[idx])

    return y
