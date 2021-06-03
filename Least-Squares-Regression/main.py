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

# create sample
sample_size = 50
xmin, xmax = -3, 3

x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)

# shuffle data
p = np.random.permutation(sample_size)
x = x[p]
y = y[p]

group_size = 10
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

            # calculate design matrix
            k_train = calc_design_matrix(x_train, x_train, h)

            # solve the least square problem
            theta = np.linalg.solve(
                k_train.T.dot(k_train) + l * np.identity(len(k_train)),
                k_train.T.dot(y_train[:, None]))

            # calculate square error
            k_test = calc_design_matrix(x_train, x_test, h)

            y_error = k_test.dot(theta) - y_test[:, None]
            error += 0.5 * (np.sum(y_error * y_error))

        error /= (sample_size // group_size)

        # create data to visualize the prediction
        k = calc_design_matrix(x, x, h)
        theta = np.linalg.solve(
            k.T.dot(k) + l * np.identity(len(k)),
            k.T.dot(y[:, None]))
        X = np.linspace(start=xmin, stop=xmax, num=5000)
        K = calc_design_matrix(x, X, h)
        prediction = K.dot(theta)

        # visualization
        axes[hi, li].set_xticks(np.arange(xmin, xmax+1))
        axes[hi, li].text(1.6, 0.95, '{:.3f}'.format(error), size=15)
        axes[hi, li].plot(X, f(X), c='red', label="ground truth")
        axes[hi, li].plot(X, prediction, c='green', label="prediction")
        axes[hi, li].scatter(x, y, c='blue', marker='o', label="sample")

for hi, h in enumerate(hs):
    axes[hi, 0].set_ylabel('h = '+'{:.0e}'.format(h), fontsize=15)

for li, l in enumerate(ls):
    axes[len(hs)-1, li].set_xlabel('l = '+'{:.0e}'.format(l), fontsize=15)

axes[0][(li // 2)].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02,), borderaxespad=1, ncol=3, fontsize=15)

plt.savefig('output.png')
