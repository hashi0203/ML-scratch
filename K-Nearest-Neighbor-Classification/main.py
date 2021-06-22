import numpy as np
from scipy.io import loadmat
from matplotlib.table import Table
import matplotlib.pyplot as plt

def knn(train_x, train_y, test_x, k_list):
    train_x = train_x.astype(np.float32).T
    test_x = test_x.astype(np.float32).T
    dist_matrix = np.sqrt(np.sum((train_x[None] - test_x[:, None]) ** 2, axis=2))
    sorted_index_matrix = np.argsort(dist_matrix, axis=1)
    ret_matrix = None
    for k in k_list:
        knn_label = train_y[sorted_index_matrix[:, :k]]
        label_sum_matrix = None
        for i in range(10):
            predict = np.sum(np.where(knn_label == i, 1, 0), axis=1)[:, None]
            if label_sum_matrix is None:
                label_sum_matrix = predict
            else:
                label_sum_matrix = np.concatenate([label_sum_matrix, predict], axis=1)
        if ret_matrix is None:
            ret_matrix = np.argmax(label_sum_matrix, axis=1)[None]
        else:
            ret_matrix = np.concatenate([ret_matrix, np.argmax(
                label_sum_matrix, axis=1)[None]], axis=0)
    return ret_matrix  # ret_matrix.shape == (len(k_list), len(test_x))

def cross_validate(train_x,train_y,t,k_list):
    Tx = np.array_split(train_x, t, 1)
    Ty = np.array_split(train_y, t)
    idx = 0
    error_rate = 0
    for i in range(t):
        l = len(Tx[i][0])
        test_y = knn(np.append(train_x[:,0:idx],train_x[:,idx+l:],axis=1),np.append(train_y[0:idx],train_y[idx+l:]),Tx[i],k_list)
        error_rate += np.count_nonzero(Ty[i] != test_y, axis=1)/l
        idx += l
    return error_rate/t

data = loadmat('digit.mat')
train_x = data['X']
test_x = data['T']

pixel = len(train_x)
train_digits = len(train_x[0])
test_digits = len(test_x[0])
digits = len(train_x[0][0])

train_x = train_x.reshape(pixel,train_digits*digits,order='F')
train_y = np.array([(i+1)%10 for i in range(digits) for _ in range(train_digits)])
test_x = test_x.reshape(pixel,len(test_x[0])*digits,order='F')
test_y = np.array([(i+1)%10 for i in range(digits) for _ in range(test_digits)])

p = np.random.permutation(train_digits*digits)
train_x = train_x[:,p]
train_y = train_y[p]

t = 20
k_list = np.arange(1,11)

cross_error = cross_validate(train_x,train_y,t,k_list)
test_error = np.count_nonzero(knn(train_x,train_y,test_x,k_list) != test_y, axis=1)/(digits*test_digits)
print('error rate (cross validation):',cross_error)
print('error rate (test):',test_error)

plt.figure()
plt.title('relationship between k and error rate')
plt.xlabel('k')
plt.ylabel('error rate')
plt.plot(k_list,cross_error,label='cross validation')
plt.plot(k_list,test_error,label='test')
plt.grid()
plt.legend()
plt.savefig("output-cv.png")

# k = 1 の時の識別結果を計算
expect = knn(train_x,train_y,test_x,[1]).reshape(10, -1)
confusion_matrix = np.array([np.array([np.sum(expect[j] == (i+1)%10) for i in range(digits)]) for j in range(digits)])

# accuracy を求める
ok = np.trace(confusion_matrix)
tot = np.sum(confusion_matrix)
accuracy = ok / tot
print("accuracy: %.4f = %d / %d" % (accuracy, ok, tot))

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
plt.text(0.5, -0.05, "accuracy: %.4f = %d / %d" % (accuracy, ok, tot), horizontalalignment="center", verticalalignment="top")

plt.savefig('output.png')



