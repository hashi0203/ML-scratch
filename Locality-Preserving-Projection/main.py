import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

np.random.seed(1)

def data_generation(i, n=100):
    assert 1 <= i <= 3
    if i == 1:
        x = np.concatenate([
                np.random.randn(n, 1) * 2, np.random.randn(n, 1)
            ], axis=1
        )
    elif i == 2:
        x = np.concatenate([
                np.random.randn(n, 1) * 2,
                2 * np.round(np.random.rand(n, 1)) - 1 + np.random.randn(n, 1) / 3.
            ], axis=1
        )
    elif i == 3:
        x = np.concatenate([
                np.random.randn(n, 1) * 2,
                4 * np.round(np.random.rand(n, 1)) - 2 + np.random.randn(n, 1) / 3.
            ], axis=1
        )
    return x

def lpp(x, n_components):
    w = np.exp(-np.sum((x[:, np.newaxis] - x[np.newaxis, :]) ** 2, axis=2))
    d = np.diag(np.sum(w, axis=1))
    l = d - w
    A = x.T.dot(l).dot(x)
    B = x.T.dot(d).dot(x)
    v, xi = scipy.linalg.eig(A, B)
    return xi[:, np.argsort(v)[:n_components]]

def visualize(x, t, name):
    plt.clf()
    plt.plot(x[:, 0], x[:, 1], 'rx')
    plt.xlim(-5., 5.)
    plt.ylim(-5., 5.)
    plt.xlabel('$x^{(1)}$')
    plt.ylabel('$x^{(2)}$')
    plt.title(name)
    a = 5. / np.max(np.abs(t), axis=0)
    for i in range(t.shape[1]):
        plt.plot(np.array([-t[0, i], t[0, i]]) * a[i], np.array([-t[1, i], t[1, i]]) * a[i], label="best-%d" % (i+1))
    plt.legend()
    plt.savefig('output-{}.png'.format(name))

n = 100
n_components = 2
for i in range(1, 4):
    x = data_generation(i, n)
    t = lpp(x, n_components)
    visualize(x, t, 'data%d' % i)


