import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
from scipy.io import loadmat

# 画像の読み込み
data = loadmat('digit.mat')
train = data['X']
test = data['T']

print("Train data: {}".format(train.shape))
print("Test data:  {}".format(test.shape))

# parameters
class_size = train.shape[1]
class_num = train.shape[2]
h = 0.5
l = 0.01

# d のみ +1 で残りが -1 の y を作成する
def generate_train_data(d):
    y = np.concatenate([-np.ones(class_size*d), np.ones(class_size), -np.ones(class_size*(class_num-d-1))])
    return y

def build_design_mat(x1, x2, bandwidth):
    return np.exp(-np.sum((x1[:, None] - x2[None]) ** 2, axis=-1) / (2 * bandwidth ** 2))

def optimize_param(design_mat, y, regularizer):
    return np.linalg.solve(
        design_mat.T.dot(design_mat) + regularizer * np.identity(len(y)),
        design_mat.T.dot(y))

def build_confusion_matrix(train_data, data, theta):
    confusion_matrix = np.zeros((class_num, class_num), dtype=np.int64)
    for i in range(class_num):
        test_data = np.transpose(data[:, :, i], (1, 0))
        design_mat = build_design_mat(train_data, test_data, h)
        prediction = []
        # 各1対他のスコアを求める
        for j in range(class_num):
            prediction += [design_mat.T.dot(theta[j])]
        # スコアが最大になる index を求める
        result = np.argmax(np.array(prediction), axis=0)
        for j in range(class_num):
            confusion_matrix[i][j] = np.sum(np.where(result == j, 1, 0))
    return confusion_matrix

x = np.concatenate(data['X'].transpose(2, 0, 1), axis=1).T

theta = []
design_mat = build_design_mat(x, x, h)
# 各1対他で theta を求める
for d in range(class_num):
    y = generate_train_data(d)
    theta += [optimize_param(design_mat, y, l)]

# 縦軸が正解のカテゴリ，横軸が予測したカテゴリの表を作る
confusion_matrix = build_confusion_matrix(x, data['T'], theta)
print('confusion matrix:')
print(confusion_matrix)

# accuracy を求める
ok = np.trace(confusion_matrix)
tot = np.sum(confusion_matrix)
accuracy = ok / tot
print("accuracy: %.4f = % d / % d" % (accuracy, ok, tot))

labels = np.append(np.arange(1, 10), 0)

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
plt.text(0.5, -0.05, "accuracy: %.4f = % d / % d" % (accuracy, ok, tot), horizontalalignment="center", verticalalignment="top")

plt.savefig('output.png')
