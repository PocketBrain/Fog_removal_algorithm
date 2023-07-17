from itertools import combinations_with_replacement
from collections import defaultdict

import numpy as np
from numpy.linalg import inv

R, G, B = 0, 1, 2


def filter(image, r):
    w, h = image.shape
    mask = np.zeros((w, h))

    # cumulative sum over Y axis
    sum_y = np.cumsum(image, axis=0)
    # difference over Y axis
    mask[:r + 1] = sum_y[r: 2 * r + 1]
    mask[r + 1:w - r] = sum_y[2 * r + 1:] - sum_y[:w - 2 * r - 1]
    mask[-r:] = np.tile(sum_y[-1], (r, 1)) - sum_y[w - 2 * r - 1:w - r - 1]

    # cumulative sum over X axis
    sum_x = np.cumsum(mask, axis=1)
    # difference over Y axis
    mask[:, :r + 1] = sum_x[:, r:2 * r + 1]
    mask[:, r + 1:h - r] = sum_x[:, 2 * r + 1:] - sum_x[:, :h - 2 * r - 1]
    mask[:, -r:] = np.tile(sum_x[:, -1][:, None], (1, r)) - \
                   sum_x[:, h - 2 * r - 1:h - r - 1]

    return mask


def real_filter(image, p, r=40, eps=1e-3):
    w, h = p.shape
    base = filter(np.ones((w, h)), r)

    means = [filter(image[:, :, i], r) / base for i in range(3)]

    mean_p = filter(p, r) / base

    means_ip = [filter(image[:, :, i] * p, r) / base for i in range(3)]

    cov_ip = [means_ip[i] - means[i] * mean_p for i in range(3)]

    var = defaultdict(dict)
    for i, j in combinations_with_replacement(range(3), 2):
        var[i][j] = filter(
            image[:, :, i] * image[:, :, j], r) / base - means[i] * means[j]

    a = np.zeros((w, h, 3))
    for y, x in np.ndindex(w, h):
        Sigmoid = np.array([[var[R][R][y, x], var[R][G][y, x], var[R][B][y, x]],
                            [var[R][G][y, x], var[G][G][y, x], var[G][B][y, x]],
                            [var[R][B][y, x], var[G][B][y, x], var[B][B][y, x]]])
        cov = np.array([c[y, x] for c in cov_ip])
        a[y, x] = np.dot(cov, inv(Sigmoid + eps * np.eye(3)))  # eq 14

    b = mean_p - a[:, :, R] * means[R] - \
        a[:, :, G] * means[G] - a[:, :, B] * means[B]

    q = (filter(a[:, :, R], r) * image[:, :, R] + filter(a[:, :, G], r) *
         image[:, :, G] + filter(a[:, :, B], r) * image[:, :, B] + filter(b, r)) / base

    return q
