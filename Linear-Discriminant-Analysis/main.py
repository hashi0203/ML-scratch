import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
from scipy.io import loadmat

data = loadmat('digit.mat')
train = data['X']
test = data['T']
# [0,...,9]の順に並び替える
train = np.append(train[:,:,9:],train[:,:,:9],axis=2)
test = np.append(test[:,:,9:],test[:,:,:9],axis=2)

print("Train data: {}".format(train.shape))
print("Test data:  {}".format(test.shape))

data_num = 10
mu = np.array([np.mean(train[:,:,i], axis=1) for i in range(data_num)])
S = np.sum(np.array([np.cov(train[:,:,i]) for i in range(data_num)]),axis=0)/data_num
invS = np.linalg.inv(S)


# log n_y は同じだから無視
# 各テストデータが何と判定されたかを計算
expect = np.array([np.argmax(np.array([mu[i][None, :].dot(invS).dot(test[:, :, j]) - mu[i][None, :].dot(invS).dot(mu[i][:, None]) / 2 for i in range(data_num)]),axis=0)[0] for j in range(data_num)])
confusion_matrix = np.array([np.array([np.sum(expect[j] == i) for i in range(data_num)]) for j in range(data_num)])

# accuracy を求める
ok = np.trace(confusion_matrix)
tot = np.sum(confusion_matrix)
accuracy = ok / tot
print("accuracy: %.4f = %d / %d" % (accuracy, ok, tot))

labels = np.arange(10)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_axis_off()
tb = Table(ax, bbox=[0,0,1,1])
width, height = 1.0 / 10, 1.0 / 10

# セル追加
for (i,j), val in np.ndenumerate(confusion_matrix):
    color = 'yellow' if i == j else 'white'
    tb.add_cell(i, j, width, height, text=val,
                loc='center', facecolor=color)

# 行ラベル
for i, label in enumerate(labels):
    tb.add_cell(i, -1, width, height, text=label, loc='right',
                edgecolor='none', facecolor='none')

# 列ラベル
for j, label in enumerate(labels):
    tb.add_cell(-1, j, width, height/2, text=label, loc='center',
                edgecolor='none', facecolor='none')

ax.add_table(tb)
plt.text(0.5, -0.05, "accuracy: %.4f = %d / %d" % (accuracy, ok, tot), horizontalalignment="center", verticalalignment="top")

plt.savefig('output.png')
