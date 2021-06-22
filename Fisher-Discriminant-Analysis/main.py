import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

np.random.seed(46)

def generate_data(sample_size=100, n_clusters=2):
    if n_clusters not in [2, 3]:
        raise ValueError('Number of clusters must be one of [2, 3].')
    x = np.random.normal(size=(sample_size, 2))
    if n_clusters == 2:
        x[:sample_size // 2, 0] -= 4
        x[sample_size // 2:, 0] += 4
    else:
        x[:sample_size // 4, 0] -= 4
        x[sample_size // 4:sample_size // 2, 0] += 4
    y = np.ones(sample_size, dtype=np.int64)
    y[sample_size // 2:] = 2
    return x, y


def fda(x, y, n_components):
    """Fisher Discriminant Analysis.
    Returns
    -------
    T : (1, 2) ndarray
        The embedding matrix.
    """

    C = x.T.dot(x)
    c, n = np.unique(y, return_counts=True)
    xy = [x[y==i] for i in c]
    mu = [np.sum(xy[i], axis=0) / n[i] for i in range(len(c))]
    Sb = np.sum(np.array([n[i] * mu[i].reshape(-1, 1).dot(mu[i].reshape(1, -1)) for i in range(len(c))]), axis=0)
    Sw = C - Sb
    v, xi = scipy.linalg.eig(Sb, Sw)

    return xi[:, np.argsort(-v)[:n_components]]


def visualize(x, y, t, name):
    plt.figure(1, (6, 6))
    plt.clf()
    plt.xlim(-7., 7.)
    plt.ylim(-7., 7.)
    plt.plot(x[y == 1, 0], x[y == 1, 1], 'bo', label='class-1')
    plt.plot(x[y == 2, 0], x[y == 2, 1], 'rx', label='class-2')
    a = 7. / np.max(np.abs(t), axis=0)
    for i in range(t.shape[1]):
        plt.plot(np.array([-t[0, i], t[0, i]]) * a[i], np.array([-t[1, i], t[1, i]]) * a[i], label="best-%d" % (i+1))
    plt.legend()
    plt.savefig('output-%s.png' % name)

n_components = 1
sample_size = 100
for n_clusters in range(2, 4):
    x, y = generate_data(sample_size=sample_size, n_clusters=n_clusters)
    t = fda(x, y, n_components)
    visualize(x, y, t, 'data%d' % n_clusters)
