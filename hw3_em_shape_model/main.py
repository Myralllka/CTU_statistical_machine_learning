import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Qt5Agg')


def estimate_shape(img, eta_1, eta_2):
    """

    :param img:
    :param eta_1:
    :param eta_2:
    :return:
    """
    s_1 = img * eta_1 - np.log(1 + np.exp(eta_1))
    s_2 = img * eta_2 - np.log(1 + np.exp(eta_2))
    return s_1 > s_2


def estimate_etas(img: np.array, s: np.array):
    """

    :param img:
    :param s:
    :return:
    """
    sum_s1 = np.sum(img[s])
    sum_s2 = np.sum(img[np.logical_not(s)])

    n1 = s.sum()
    n2 = np.logical_not(s).sum()

    eta_1 = np.log(sum_s1 / (n1 - sum_s1))
    eta_2 = np.log(sum_s2 / (n2 - sum_s2))
    return tuple((eta_1, eta_2))


def shape_mle(avg_image: np.array, etas_init: tuple):
    """
    returns a shape s and a 2-tuple η.
    :param avg_image:  the average image
    :param etas_init: - a 2-tuple representing an initial estimate of the Bernoulli distribution parameters
    :return: 
    """
    s_new = np.zeros(avg_image.shape, dtype=bool)
    etas_new = etas_init
    while True:
        s_prev = s_new
        s_new = estimate_shape(avg_image, etas_new[0], etas_new[1])
        if np.all(s_new == s_prev):
            break
        etas_new = estimate_etas(avg_image, s_new)
        print(".")

    return s_new, etas_new


if __name__ == "__main__":
    # data - batches * height * width
    data = np.load("images0.npy")
    data.shape[0]
    data_avg = data.sum(axis=0) / data.shape[0]
    s, etas = shape_mle(data_avg, tuple((-1, 1)))
    print(etas)
    plt.imshow(s)
    plt.show()
    # for im in data:
    #     plt.imshow(im)
    #     plt.show()
