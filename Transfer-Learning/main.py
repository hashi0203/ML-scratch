import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def generate_data(n_total, n_positive):
    x = np.random.normal(size=(n_total, 2))
    x[:n_positive, 0] -= 2
    x[n_positive:, 0] += 2
    x[:, 1] *= 2.
    y = np.empty(n_total, dtype=np.int64)
    y[:n_positive] = 0
    y[n_positive:] = 1
    return x, y

def build_design_mat(x1, x2, h):
    return np.exp(-np.sum((x1[:, None] - x2[None]) ** 2, axis=2) / (2 * h ** 2))

def cwls(train_x, train_y, test_x):
    n = len(test_x)
    nyp = np.sum(train_y==1)
    nyn = np.sum(train_y==0)
    xp = train_x[train_y==1]
    xn = train_x[train_y==0]
    app = 1 / (nyp * nyp) * np.sum(np.sqrt(np.sum((xp[:, np.newaxis] - xp[np.newaxis, :]) ** 2, axis=2)))
    apn = 1 / (nyp * nyn) * np.sum(np.sqrt(np.sum((xp[:, np.newaxis] - xn[np.newaxis, :]) ** 2, axis=2)))
    ann = 1 / (nyn * nyn) * np.sum(np.sqrt(np.sum((xn[:, np.newaxis] - xn[np.newaxis, :]) ** 2, axis=2)))
    bp = 1 / (n * nyp) * np.sum(np.sqrt(np.sum((test_x[:, np.newaxis] - xp[np.newaxis, :]) ** 2, axis=2)))
    bn = 1 / (n * nyn) * np.sum(np.sqrt(np.sum((test_x[:, np.newaxis] - xn[np.newaxis, :]) ** 2, axis=2)))

    pi = (apn - ann - bp + bn) / (2 * apn - app - ann)
    pi = min(1, max(0, pi))

    W = np.diag(np.array([(1-pi) / nyn, pi / nyp])[train_y] * len(train_y))
    phi = np.hstack((np.ones(len(train_x)).reshape(-1,1), train_x))
    theta = np.linalg.solve(
        phi.T.dot(W).dot(phi), phi.T.dot(W).dot(train_y)
    )

    return theta

def visualize(train_x, train_y, test_x, test_y, theta):
    for x, y, name in [(train_x, train_y, 'train'), (test_x, test_y, 'test')]:
        plt.clf()
        plt.figure(figsize=(6, 6))
        plt.xlim(-5., 5.)
        plt.ylim(-7., 7.)
        lin = np.array([-5., 5.])
        plt.plot(lin, -(theta[2] + lin * theta[0]) / theta[1])
        plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1],
                    marker='$O$', c='blue')
        plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1],
                    marker='$X$', c='red')
        plt.title(name)
        plt.savefig('output-{}.png'.format(name))

train_x, train_y = generate_data(n_total=100, n_positive=90)
eval_x, eval_y = generate_data(n_total=100, n_positive=10)
theta = cwls(train_x, train_y, eval_x)
visualize(train_x, train_y, eval_x, eval_y, theta)
