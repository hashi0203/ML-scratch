import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

import os
import argparse
import datetime

import functools
print = functools.partial(print, flush=True)

num_epochs = 500
num_classes = 10

decay = 1e-5
dr = 0.5
smoothing = 0.1
flood = 0.03

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--early-stopping', '-e', action='store_true', help='use early stopping')
parser.add_argument('--weight-decay', '-w', action='store_true', help='use weight decay')
parser.add_argument('--dropout', '-d', action='store_true', help='use dropout')
parser.add_argument('--label-smoothing', '-l', action='store_true', help='use label smoothing')
parser.add_argument('--flooding', '-f', action='store_true', help='use flooding')
# --summary, -s オプションで torchsummary を表示するかを指定
parser.add_argument('--summary', '-s', action='store_true', help='show torchsummary')
args = parser.parse_args()

e = 'e' if args.early_stopping else '-'
w = 'w' if args.weight_decay else '-'
d = 'd' if args.dropout else '-'
l = 'l' if args.label_smoothing else '-'
f = 'f' if args.flooding else '-'
reg = '%s%s%s%s%s' % (e, w, d, l, f)

# 重み減衰
decay = decay if args.weight_decay else 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 変換器の作成
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # torch.Tensor へ変換
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # 正規化する
])
transform_test = transforms.Compose([
    transforms.ToTensor(), # torch.Tensor へ変換
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # 正規化する
])

# 訓練データのダウンロードと変換
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# 訓練データのローダ(読み込み器)の作成
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# テストデータのダウンロードと変換
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# テストデータのローダ(読み込み器)の作成
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


# ラベル平滑化
class SmoothCrossEntropyLoss(_WeightedLoss):
    global num_classes, smoothing

    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        with torch.no_grad():
            targets = F.one_hot(targets, num_classes)
            targets = targets * (1-smoothing) + smoothing / num_classes
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

# ボトルネックブロックの作成
class Bottleneck(nn.Module):
    expansion = 4

    def __init__ (self, ich, mch, stride=1):
        super(Bottleneck, self).__init__()
        och = mch * self.expansion

        # 1 x 1
        self.conv1 = nn.Conv2d(ich, mch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mch)

        # 3 x 3
        self.conv2 = nn.Conv2d(mch, mch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mch)

        self.dr = nn.Dropout(dr) if args.dropout else lambda x : x # ドロップアウト

        # 1 x 1
        self.conv3 = nn.Conv2d(mch, och, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(och)

        self.shortcut = nn.Sequential()
        if stride != 1 or ich != och:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ich, och, 1, stride=stride, bias=False),
                nn.BatchNorm2d(och)
            )

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.dr(h) # ドロップアウト
        h = self.conv3(h)
        h = self.bn3(h)
        h += self.shortcut(x)
        h = F.relu(h)
        return h

# ResNet の作成
class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.ich = 64

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.bottleneck1 = self._make_layer(64, num_blocks[0])
        self.bottleneck2 = self._make_layer(128, num_blocks[1], stride=2)
        self.bottleneck3 = self._make_layer(256, num_blocks[2], stride=2)
        self.bottleneck4 = self._make_layer(512, num_blocks[3], stride=2)

        self.fc = nn.Linear(512*Bottleneck.expansion, num_classes)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)

        h = self.bottleneck1(h)
        h = self.bottleneck2(h)
        h = self.bottleneck3(h)
        h = self.bottleneck4(h)

        h = F.avg_pool2d(h, h.size()[2:])
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        return h

    def _make_layer(self, mch, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.ich, mch, stride))
            self.ich = mch * Bottleneck.expansion
        return nn.Sequential(*layers)

def ResNet50(num_classes):
    return ResNet([3, 4, 6, 3], num_classes)

model = ResNet50(num_classes).to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model) # DataParallel を使って高速化
    cudnn.benchmark = True

if args.summary:
    summary(model, (3, 32, 32))

criterion = nn.CrossEntropyLoss() if not(args.label_smoothing) else SmoothCrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=decay)

# 学習
train_loss = []
train_acc = []
test_loss = []
test_acc = []

def train():
    global trainloader, model, criterion, optimizer, args

    model.train()
    sum_loss = 0
    correct = 0
    total = 0

    for (inputs, targets) in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        # 勾配を0に初期化する（逆伝播に備える）．
        optimizer.zero_grad()
        # 順伝播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if args.flooding: loss = abs(loss-flood) + flood # 洪水
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
            if args.flooding: loss = abs(loss-flood) + flood

            sum_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    return sum_loss / total, 100 * correct / total

def visualize():
    global num_epochs, train_loss, train_acc, test_loss, test_acc, saved_epoch
    global LOSS_FILE, ACC_FILE

    epochs = np.arange(1, num_epochs+1)

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
    plt.savefig(LOSS_FILE)

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

    plt.plot(saved_epoch, test_acc[saved_epoch-1], color='tab:orange', marker='o')
    if saved_epoch != epochs[am]:
        plt.text(saved_epoch, test_acc[saved_epoch-1]+0.3, '%.3f' % test_acc[saved_epoch-1], horizontalalignment="center", verticalalignment="bottom")

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend()
    plt.title('accuracy')
    plt.savefig(ACC_FILE)

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
if not os.path.isdir('graph'):
    os.mkdir('graph')

t = datetime.datetime.now().strftime('%m%d-%H%M')
CKPT_FILE = './checkpoint/ckpt-%s-%s.pth' % (reg, t)
LOSS_FILE = './graph/loss-%s-%s.png' % (reg, t)
ACC_FILE = './graph/accuracy-%s-%s.png' % (reg, t)

saved_epoch, saved_acc = 0, 0

for epoch in range(1, num_epochs+1):
    # 訓練
    loss, acc = train()
    train_loss += [loss]
    train_acc += [acc]

    # テスト
    loss, acc = test()
    test_loss += [loss]
    test_acc += [acc]

    print('epoch %2d | train_loss: %.3f, train_acc: %.2f %%, test_loss: %.3f, test_acc: %.2f %%'
        % (epoch, train_loss[-1], train_acc[-1], test_loss[-1], test_acc[-1]))

    # checkpoint の保存
    if args.early_stopping:
        # 早期終了の時は精度が最も高くなったときに保存
        if test_acc[-1] > saved_acc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': test_acc[-1],
                'epoch': epoch,
            }
            torch.save(state, CKPT_FILE)
            saved_epoch, saved_acc = epoch, test_acc[-1]
    else:
        # 早期終了でない時は最後に保存
        if epoch == num_epochs:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': test_acc[-1],
                'epoch': epoch,
            }
            torch.save(state, CKPT_FILE)
            saved_epoch, saved_acc = epoch, test_acc[-1]

visualize()