import numpy as np
import torch
import copy
from utils import OneHot



class AutomaticDataEnhancement:

    '''
    # Best till now is
    # 100, 50, 10**-10 -> 145/150 bei p=0.4
    # 100, 20, 10**-10
    # 50, 50, 1e-5 -> 147/150 bei p=0.4

    Implements Automatic Data Enhancement short ADE from paper in References.
    Automatically corrects labels.

    Parameters:
    -----------

    N_epoch: {int}, default = 50
        Number of learning steps per epoch.

    N_era: {int}, default = 50
        Maximal number of eras. Early stopping if convergence.

    learning_rate_w: {float}, default = 1e-5
        Learning rate of the MLP parameters.

    learning_rate_p: {float}, default = 1e-3
        Learning rate of the class probabilities adaption.

    eps: {float}, default = 1e-5
        Convergence criterion w.r.t. era time


    Attributes:
    -----------

    .winner_H: {int}
        number of hidden nodes that lead to lowest total loss

    .memory_SSE: {np.array of lists} of quasi-shape = (max_number_of_hidden_nodes, steps_till_convergence)
        Loss for different number of hidden nodes at different era times.


    Methods:
    -----------

    .fit_transform(X, y):

        Parameters:
        -----------

        X: {np.array} of shape (n_samples, n_features)

        y: {np.array} of shape (n_samples)

        Returns: Corrected labels


    References:
    ------------

        [1] An algorithm for correcting mislabeled data. Xinchuan Zeng and Tony R. Martinez.
            Computer Science Department, Brigham Young University, Provo, UT 84602, USA


    Example:
    -----------
    from ALC import AutomaticDataEnhancement as ADE
    from ALC.utils import OneHot

    ade = ADE()
    onehot = OneHot()

    y = onehot.encode(y)
    y_corrected = ade.fit_transform(X, y)
    y_corrected = onehot.decode(y_corrected)


    '''

    def __init__(self, N_epoch = 50, N_era = 50, learning_rate_w = 1e-5, learning_rate_p = 1e-3, eps = 1e-5):

        self.N_epoch = N_epoch
        self.N_era = N_era
        self.learning_rate_w = learning_rate_w
        self.learning_rate_p = learning_rate_p
        self.memory_SSE = None
        self.winner_H = None
        self.eps = eps


    def fit_transform(self, X, y):

        onehot = OneHot()

        # Transform data into one Hot notation
        y = onehot.encode(y)

        self.X = X
        self.class_ratio_init = self._get_class_ratio_of(y)

        self.N = X.shape[0]
        self.D_in = X.shape[1]
        self.D_out = y.shape[1]

        hidden_nodes = 0
        self.H = hidden_nodes

        memory_labels = []
        memory_SSE_adj = []

        while True:

            # Initialise adjustable set of labels and label probabilities
            self.y_adj, self.p = self._initialize(y)

            # Parameterize model
            hidden_nodes += 1
            self.H += 1
            memory_SSE_adj.append([])

            self.model = MLP(self.D_in, hidden_nodes, self.D_out, self.learning_rate_w)

            for current_era in range(self.N_era):
                mem = memory_SSE_adj[-1]

                _ = self._era()
                mem.append(_)

                if len(mem) > 2 and np.abs(mem[-1] - mem[-2]) < self.eps:
                    memory_labels.append(self.y_adj)
                    break

                if current_era == (self.N_era-1):
                    memory_labels.append(self.y_adj)
                    print('Did not reach convergence target for %d hidden nodes!' % hidden_nodes)


            if (hidden_nodes >= 3) and (memory_SSE_adj[-1][-1] > memory_SSE_adj[-2][-1]) and (memory_SSE_adj[-2][-1] > memory_SSE_adj[-3][-1]):
                break

            if hidden_nodes > 10:
                print('Early stopping even though no convergence yet')
                break

        self.memory_SSE = np.array(memory_SSE_adj)

        lowest_loss_of_hidden_nodes = np.array([self.memory_SSE[x][-1] for x in range(self.memory_SSE.shape[0])])
        self.winner_H = np.argmin(lowest_loss_of_hidden_nodes)+1

        return onehot.decode(memory_labels[self.winner_H -1])


    def _initialize(self, y):

        p = copy.copy(y)

        return p, np.where(y==1, 0.95, 0.05/(self.D_out-1))


    def _epoch(self):

        # Update NN weights
        self.model.do_step(self.X, self.y_adj)

        # Obtain updated class prob
        p_target = self.model(self.X)

        # Update p
        self.p += self.learning_rate_p*(p_target - self.p)

        # Update labels
        self.y_adj[...] = 0
        self.y_adj[np.arange(self.p.shape[0]), np.argmax(self.p, axis = -1)] = 1


    def _era(self):

        # One era consits of N_epoch epochs, and then calculating an extended Loss

        # Do epochs
        for _ in range(self.N_epoch):
            self._epoch()

        ## Calculate loss from deviating class ratios
        # Current class ratio
        current_class_ratio = self._get_class_ratio_of(self.y_adj)

        loss_dist = self._loss_dist(current_class_ratio, self.class_ratio_init)


        ## Calculate extended loss
        loss = self.model.current_loss + loss_dist + self._loss_hn()
        return loss

    def _get_class_ratio_of(self, y):
        return y.sum(axis = 0).astype(np.float32)/y.shape[0]

    def _loss_dist(self, q_curr, q_init):
        C = self.D_out
        D = np.abs(q_curr - q_init)/q_init
        B = np.where(D < 0.05, 0.1, 1.0)

        return self.N*(C - 1)/C*(B*D*np.where(q_curr > q_init, q_curr, q_init)).sum()

    def _loss_hn(self):
        A_1 = 0.05
        A_2 = 0.2
        C = self.D_out
        I = self.D_in
        H = self.H

        if H <= I:
            return A_1*(H-1)*self.N*(C-1)/C
        else:
            return (A_1*(I-1) + A_2*(H-I))*self.N*(C-1)/C


class MLP:

    def __init__(self, D_in, H, D_out, learning_rate_w):

        #self.learning_rate_w = learning_rate_w

        self.model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
            torch.nn.Softmax(dim = -1),
        )

        self.loss_fn = torch.nn.MSELoss(reduction='sum')

        self.optim = torch.optim.SGD(self.model.parameters(), lr = learning_rate_w, momentum=0.9)

    def __call__(self, X):
        X = torch.Tensor(X)
        return self.model(X).detach().numpy()

    def do_step(self, X, y):
        X = torch.Tensor(X)
        y = torch.Tensor(y)

        # Predict targets
        y_pred = self.model(X)

        # Calculate SSE
        loss = self.loss_fn(y_pred, y)
        self.current_loss = loss.detach().numpy().tolist()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
