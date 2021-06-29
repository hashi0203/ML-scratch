import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

# MNIST データセットの用意
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 多層パーセプトロン
class MLP(nn.Module):
    # 3層ニューラルネットワーク
    # 隠れ層のニューロンはそれぞれ 1024，512
    # 活性化関数は ReLU
    # 活性化関数の後に dropout を挿入

    in_dim = 784
    hid1_dim = 1024
    hid2_dim = 512
    out_dim = 10

    def __init__(self):
        super(MLP, self).__init__()

        fcs = []
        fcs.append(nn.Linear(self.in_dim, self.hid1_dim))
        fcs.append(nn.ReLU())
        fcs.append(nn.Dropout(0.2))
        fcs.append(nn.Linear(self.hid1_dim, self.hid2_dim))
        fcs.append(nn.ReLU())
        fcs.append(nn.Dropout(0.2))
        fcs.append(nn.Linear(self.hid2_dim, self.out_dim))
        self.fcs = nn.Sequential(*fcs)

    def forward(self, x):
        x = x.view(-1, self.in_dim)
        return self.fcs(x)


model = MLP().to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model) # DataParallel を使って高速化
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 学習
n_epoch = 70

train_loss = []
train_acc = []
test_loss = []
test_acc = []

def train():
    global trainloader, model, criterion, optimizer

    model.train()
    sum_loss = 0
    correct = 0
    total = 0

    for _, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # 勾配を0に初期化する（逆伝播に備える）．
        optimizer.zero_grad()
        # 順伝播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # 逆伝播
        loss.backward()
        # 重みの更新
        optimizer.step()

        sum_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    return sum_loss / total, 100 * correct / total

def test():
    global testloader, model, criterion, optimizer

    model.eval()
    sum_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for (inputs, targets) in testloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            sum_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    return sum_loss / total, 100 * correct / total

def visualize():
    global n_epoch, train_loss, train_acc, test_loss, test_acc

    epochs = np.arange(1, n_epoch+1)

    # loss の可視化
    plt.figure()

    plt.plot(epochs, train_loss, label="train", color='tab:blue')
    am = np.argmin(train_loss)
    plt.plot(epochs[am], train_loss[am], color='tab:blue', marker='x')
    plt.text(epochs[am], train_loss[am]-0.01, '%.3f' % train_loss[am], horizontalalignment="center", verticalalignment="top")

    plt.plot(epochs, test_loss, label="test", color='tab:orange')
    am = np.argmin(test_loss)
    plt.plot(epochs[am], test_loss[am], color='tab:orange', marker='x')
    plt.text(epochs[am], test_loss[am]+0.01, '%.3f' % test_loss[am], horizontalalignment="center", verticalalignment="bottom")

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.title('loss')
    plt.savefig("loss.png")

    # accuracy の可視化
    plt.figure()

    plt.plot(epochs, train_acc, label="train", color='tab:blue')
    am = np.argmax(train_acc)
    plt.plot(epochs[am], train_acc[am], color='tab:blue', marker='x')
    plt.text(epochs[am], train_acc[am]+0.3, '%.3f' % train_acc[am], horizontalalignment="center", verticalalignment="bottom")

    plt.plot(epochs, test_acc, label="test", color='tab:orange')
    am = np.argmax(test_acc)
    plt.plot(epochs[am], test_acc[am], color='tab:orange', marker='x')
    plt.text(epochs[am], test_acc[am]-0.3, '%.3f' % test_acc[am], horizontalalignment="center", verticalalignment="top")

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend()
    plt.title('accuracy')
    plt.savefig("accuracy.png")


for epoch in range(n_epoch):
    # 訓練
    loss, acc = train()
    train_loss += [loss]
    train_acc += [acc]

    # テスト
    loss, acc = test()
    test_loss += [loss]
    test_acc += [acc]

    print('epoch %2d | train_loss: %.3f, train_acc: %.2f %%, test_loss: %.3f, test_acc: %.2f %%'
        % (epoch+1, train_loss[-1], train_acc[-1], test_loss[-1], test_acc[-1]))

visualize()