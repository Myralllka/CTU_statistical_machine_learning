import numpy as np


class SoftmaxLayer(object):
    def __init__(self, name):
        super(SoftmaxLayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        """
        Forward message.
        Information about the stability and optimization is taken from here: http://saitcelebi.com/tut/output/part2.html
        :param X: ReLU inputs (outputs of the previous layer), layer output, shape (n_samples, n_units)
        :return: delta vector from the loss layer, shape (n_samples, n_units)
        """
        exps = np.exp(X - np.max(X, 1, keepdims=True))
        denom = np.sum(exps, 1, keepdims=True)
        res = exps / denom
        assert res.shape == X.shape, "{}: wrong forward input shape".format(self.name)
        if np.isnan(res.sum()):
            print('Found Nan in {} forward'.format(self.name))
        return res

    def delta(self, Y, delta_next):
        """
        Computes delta (dl/d(layer inputs)), based on delta from the following layer. The computations involve backward
        message.
        :param Y: output of this layer (i.e., input of the next), shape (n_samples, n_units)
        :param delta_next: delta vector backpropagated from the following layer, shape (n_samples, n_units)
        :return: delta vector from this layer, shape (n_samples, n_units)
        """
        assert Y.shape == delta_next.shape, "{}: wrong delta input shape".format(self.name)
        n_samples, n_units = Y.shape
        p = Y[0].shape

        identity = np.array([np.eye(n_units) * Y[i] for i in range(n_samples)])
        reshaped = np.array([np.reshape(Y[i], (-1, 1)) @ np.reshape(Y[i], (1, -1)) for i in range(n_samples)])
        d_smax = identity - reshaped
        res = np.array([delta_next[i] @ d_smax[i] for i in range(n_samples)])

        assert res.shape == Y.shape
        if np.isnan(res.sum()):
            print('Found Nan in {} delta'.format(self.name))

        return res
