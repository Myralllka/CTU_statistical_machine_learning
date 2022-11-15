import numpy as np


class LossCrossEntropy(object):
    def __init__(self, name):
        super(LossCrossEntropy, self).__init__()
        self.name = name
        self.eps = 10e-9  # to avoid infinities in log

    def forward(self, X, T):
        """
        Forward message.
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs)
        :return: layer output, shape (n_samples, 1)
        """
        assert X.shape == T.shape, "{}: wrong shape of input".format(self.name)

        res = -np.sum(np.log(X + self.eps) * T, 1, keepdims=True)

        if np.isnan(res.sum()):
            print('Found Nan in {} forward'.format(self.name))
        return res


    def delta(self, X, T):
        """
        Computes delta vector for the output layer.
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs)
        :return: delta vector from the loss layer, shape (n_samples, n_inputs)
        """
        assert X.shape == T.shape, "{}: wrong delta input sape".format(self.name)

        res = -T / (X + self.eps)

        if np.isnan(res.sum()):
            print('Found Nan in {} delta'.format(self.name))
        return res


class LossCrossEntropyForSoftmaxLogits(object):
    def __init__(self, name):
        super(LossCrossEntropyForSoftmaxLogits, self).__init__()
        self.name = name
        self.eps = 10e-9  # to avoid infinities in log

    def forward(self, X, T):
        res = np.exp(X - np.max(X, 1, keepdims=True))
        sum_vec = res.sum(1, keepdims=True)

        res = -np.sum((X - np.log(sum_vec + self.eps)) * T, 1, keepdims=True)

        if np.isnan(res.sum()):
            print('Found Nan in {} forward'.format(self.name))
        assert res.shape == (X.shape[0], 1)

        return res

    def delta(self, X, T):
        res = np.exp(X - np.max(X, 1, keepdims=True))
        res /= np.sum(res, 1, keepdims=True)

        if np.isnan(res.sum()):
            print('Found Nan in {} delta'.format(self.name))

        assert res.shape == X.shape

        return res - T
