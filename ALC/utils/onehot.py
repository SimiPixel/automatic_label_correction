import numpy as np

class OneHot:

    def encode(self, y):
        encoded_y = np.zeros((y.size, y.max()+1))
        encoded_y[np.arange(y.size), y] = 1
        return encoded_y

    def decode(self, y):
        return np.argmax(y, axis = -1)
