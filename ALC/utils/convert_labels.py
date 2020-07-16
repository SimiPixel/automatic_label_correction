import numpy as np

def convert_labels(y, unique_labels):
    y_new_format = np.zeros(y.shape)

    count = -1
    for label in unique_labels:
        count += 1
        for idx in range(y.shape[0]):
            if y[idx] == label:
                y_new_format[idx] = count

    return y_new_format.astype(int)
