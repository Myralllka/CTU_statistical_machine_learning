import numpy as np


class LinearLayer(object):
    def __init__(self, n_inputs, n_units, rng, name):
        """
        Linear (dense, fully-connected) layer.
        :param n_inputs: Number of inputs, N
        :param n_units: Number of outputs, K
        :param rng: random number generator used for initialization
        :param name:
        """
        super(LinearLayer, self).__init__()
        self.b = None
        self.W = None
        self.n_inputs = n_inputs
        self.n_units = n_units
        self.rng = rng
        self.name = name
        self.initialize()

    def has_params(self):
        return True

    def forward(self, X):
        """
        Forward message.
        :param X: layer inputs, shape (n_samples, n_inputs)
        :return: layer output, shape (n_samples, n_units)
        """
        assert X.shape[1] == self.n_inputs, "{}: wrong forward input shape".format(self.name)
        res = (X @ self.W) + self.b

        if np.isnan(res.sum()):
            print('Found Nan in {} forward'.format(self.name))
        return res

    def delta(self, Y, delta_next):
        """
        Computes delta (dl/d(layer inputs)), based on delta from the following layer. The computations involve backward
        message.
        :param Y: output of this layer (i.e., input of the next), shape (n_samples, n_units)
        :param delta_next: delta vector backpropagated from the following layer, shape (n_samples, n_units)
        :return: delta vector from this layer, shape (n_samples, n_inputs)
        """
        assert Y.shape == delta_next.shape, "{}: wrong delta input shape".format(self.name)
        res = delta_next @ self.W.T
        if np.isnan(res.sum()):
            print('Found Nan in {} delta'.format(self.name))
        return res

    def grad(self, X, delta_next):
        """
        Gradient averaged over all samples. The computations involve parameter message.
        :param X: layer input, shape (n_samples, n_inputs)
        :param delta_next: delta vector backpropagated from the following layer, shape (n_samples, n_units)
        :return: a list of two arrays [dW, db] corresponding to gradients of loss w.r.t. weights and biases, the shapes
        of dW and db are the same as the shapes of the actual parameters (self.W, self.b)
        """
        assert X.shape[1] == self.n_inputs, "{}: wrong delta X input shape".format(self.name)
        assert delta_next.shape[1] == self.n_units, "{}: wrong delta delta_next input shape".format(self.name)
        sample_size = X.shape[0]

        dW = X.T @ delta_next
        db = np.sum(delta_next, 0)

        if np.isnan(dW.sum()):
            print('Found Nan in {} grad dW'.format(self.name))
        if db is None:
            print('Parameter b in {} grad is none'.format(self.name))

        return [dW / sample_size, db / sample_size]


    def initialize(self):
        """
        Perform He's initialization (https://arxiv.org/pdf/1502.01852.pdf). This method is tuned for ReLU activation
        function. Biases are initialized to 1 increasing probability that ReLU is not initially turned off.
        """
        scale = np.sqrt(2.0 / self.n_inputs)
        self.W = self.rng.normal(loc=0.0, scale=scale, size=(self.n_inputs, self.n_units))
        self.b = np.ones(self.n_units)

    def update_params(self, dtheta):
        """
        Updates weighs and biases.
        :param dtheta: contains a two element list of weight and bias updates the shapes of which corresponds to self.W
        and self.b
        """
        assert len(dtheta) == 2, len(dtheta)
        dW, db = dtheta
        assert dW.shape == self.W.shape, dW.shape
        assert db.shape == self.b.shape, db.shape
        self.W += dW
        self.b += db
