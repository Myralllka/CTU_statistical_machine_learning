import copy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy

mpl.use('Qt5Agg')


def estimate_pis(axrs):
    """

    :param axrs: auxiliary variables of the EM algorithm
    :param ps_old: Pi_s to be updated
    :return:
    """
    res = np.sum(axrs, axis=1) / axrs.shape[1]
    assert np.abs(np.sum(res) - 1) < 10e-9, "wrong sum"
    return res


def estimate_shape(img, eta_0, eta_1):
    """

    :param img:
    :param eta_0:
    :param eta_1:
    :return:
    """
    eps = 10e-9
    s_1 = img * eta_0 - np.log(1 + np.exp(eta_0) + eps)
    s_2 = img * eta_1 - np.log(1 + np.exp(eta_1) + eps)
    return s_1 > s_2


def estimate_etas(img: np.array, s: np.array):
    """

    :param img:
    :param s:
    :return:
    """
    eps = 10e-9

    sum_s1 = np.sum(img[s]) + eps
    sum_s2 = np.sum(img[np.logical_not(s)]) + eps

    n1 = s.sum() + eps
    n2 = np.logical_not(s).sum() + eps

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


def posterior_pose_prob(images: np.array, ets: tuple, shape: np.array, pis: np.array):
    """

    :param images: B*H*W (?) array of images
    :param ets: tuple of etas
    :param shape:
    :param pis:
    :return:
    """
    # eta(Trs)
    shape = copy.deepcopy(shape) * ets[1]
    shape = np.logical_not(shape) * ets[0]
    # rotate images
    assert np.all(np.abs(np.rot90(shape, 2) - scipy.ndimage.rotate(shape, angle=180)) < 10e-5), "error in rotation"
    s = np.array([shape,
                  np.rot90(shape, 1),
                  np.rot90(shape, 2),
                  np.rot90(shape, 3)])
    # <x, eta(Trs)>
    a = np.einsum("ijk,ljk->li", images, s)
    # res of softmax: 4 * B array
    res = scipy.special.softmax(a + np.log(pis.reshape(4, 1)), 0)
    return res


if __name__ == "__main__":
    # data - batches * height * width
    data = np.load("images.npy")

    init_s = np.random.randint(2, size=data[0].shape)
    avg_img = data.sum(axis=0) / data.shape[0]
    init_etas = tuple((-0.1, 0.1))
    init_ps = np.array([0.25, 0.25, 0.25, 0.25])
    # s, etas = shape_mle(avg_img, init_etas)
    # plt.imshow(s)
    # plt.show()
    # while True:
    #     pass

    # data4 | <R B W H>
    data4 = np.array([data,
                      np.rot90(data, -1, axes=(1, 2)),
                      np.rot90(data, -2, axes=(1, 2)),
                      np.rot90(data, -3, axes=(1, 2))])
    ets = copy.deepcopy(init_etas)
    s = copy.deepcopy(init_s)
    ps = copy.deepcopy(init_ps)
    s_prev = copy.deepcopy(s)
    cnt = 0
    # for i in range(11):
    while True:
        # axr | <R B 1 1>
        axr = posterior_pose_prob(data, ets, s, ps)
        axr = axr.reshape(*axr.shape, 1, 1)
        # # # # # # # # # # #
        # Daxr | <R B W H>  #
        Daxr = data4 * axr
        # sum over R | <B W H>
        sr = np.sum(Daxr, axis=0)
        # sum over batches | <W H>
        sx = np.sum(sr, axis=0)
        psi = sx / data.shape[0]
        # plt.imshow(psi)
        # plt.show()
        s, ets = shape_mle(psi, etas_init=ets)
        ps = estimate_pis(axr.reshape(*axr.shape[:2]))
        if np.all(s_prev == s):
            break
        s_prev = copy.deepcopy(s)
        cnt += 1
    print(cnt)
    plt.imshow(s)
    plt.show()
