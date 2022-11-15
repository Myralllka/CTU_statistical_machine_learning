import numpy as np


class ReLULayer(object):
    def __init__(self, name):
        super(ReLULayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        res = np.maximum(.0, X)
        assert res.shape == X.shape, "{}: wrong forward input shape".format(self.name)
        if np.isnan(res.sum()):
            print('Found Nan in {} forward'.format(self.name))
        return res

    def delta(self, Y, delta_next):
        res = np.array(delta_next, copy=True)
        assert res.shape == Y.shape, "{}: wrong delta input shape".format(self.name)
        res[Y <= 0] = 0
        if np.isnan(res.sum()):
            print('Found Nan in {} delta'.format(self.name))
        return res
