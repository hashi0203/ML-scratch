import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)  # set the random seed for reproducibility

def f(x):
    pix = np.pi * x
    return np.sin(pix) / pix + 0.1 * x

def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    target = f(x)
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
    return x, target + noise

def calc_design_matrix(x, c, h):
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))

def solve(x, y, data_size, h, l):
    # calculate design matrix
    k = calc_design_matrix(x, x, h)
    kki = k.dot(k) + np.identity(data_size)

    # initialize
    z = np.random.randn(data_size)
    u = np.random.randn(data_size)
    o_theta = np.zeros(data_size)

    while True:
        theta = np.linalg.solve(
            kki,
            k.dot(y) + z - u)
        z = np.maximum(0, theta + u - l) + np.minimum(0, theta + u + l)
        u = u + theta - z
        if np.sum((theta - o_theta)**2) < 1e-7:
            break
        o_theta = theta
    return z

# create sample
sample_size = 50
xmin, xmax = -3, 3

x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)

# shuffle data
p = np.random.permutation(sample_size)
x = x[p]
y = y[p]

group_size = 10
train_size = sample_size - group_size
p = np.arange(sample_size)
p = (p + group_size) % sample_size

hs = [1e-4, 1e-2, 1, 1e2]
ls = [1e-6, 1e-4, 1e-2, 1, 1e2]
fig, axes = plt.subplots(nrows=len(hs), ncols=len(ls), sharex=False, figsize=(len(ls)*4, len(hs)*4))
for hi, h in enumerate(hs):
    for li, l in enumerate(ls):
        error = 0
        for _ in range(sample_size // group_size):
            x_test = x[:group_size]
            x_train = x[group_size:]

            y_test = y[:group_size]
            y_train = y[group_size:]

            x = x[p]
            y = y[p]

            theta = solve(x_train, y_train, train_size, h, l)

            # calculate square error
            k_test = calc_design_matrix(x_train, x_test, h)

            y_error = k_test.dot(theta) - y_test
            error += 0.5 * (np.sum(y_error * y_error))

        error /= (sample_size // group_size)

        # create data to visualize the prediction
        theta = solve(x, y, sample_size, h, l)
        zeros = np.count_nonzero(abs(theta) < 1e-5)
        X = np.linspace(start=xmin, stop=xmax, num=5000)
        K = calc_design_matrix(x, X, h)
        prediction = K.dot(theta)

        # visualization
        axes[hi, li].set_xticks(np.arange(xmin, xmax+1))
        axes[hi, li].text(1.6, 0.95, '{:.3f}'.format(error), size=15)
        axes[hi, li].text(1.6, 0.80, zeros, size=15)
        axes[hi, li].plot(X, f(X), c='red', label="ground truth")
        axes[hi, li].plot(X, prediction, c='green', label="prediction")
        axes[hi, li].scatter(x, y, c='blue', marker='o', label="sample")

for hi, h in enumerate(hs):
    axes[hi, 0].set_ylabel('h = '+'{:.0e}'.format(h), fontsize=15)

for li, l in enumerate(ls):
    axes[len(hs)-1, li].set_xlabel('l = '+'{:.0e}'.format(l), fontsize=15)

axes[0][(li // 2)].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02,), borderaxespad=1, ncol=3, fontsize=15)

plt.savefig('output.png')