import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

from model import Autoencoder

import numpy as np
import matplotlib.pyplot as plt

import os
import argparse
import datetime

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
parser.add_argument('--decay', default=8e-6, type=float, help='weight decay')
# --summary, -s オプションで torchsummary を表示するかを指定
parser.add_argument('--summary', '-s', action='store_true', help='show torchsummary')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

# MNIST データセットの用意
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=2)

model = Autoencoder().to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model) # DataParallel を使って高速化
    cudnn.benchmark = True

if args.summary:
    summary(model, (1, 28, 28))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

# 学習
n_epoch = 10

train_loss = []
test_loss = []

def train():
    global trainloader, model, criterion, optimizer

    model.train()
    sum_loss = 0
    total = 0

    for (inputs, _) in trainloader:
        inputs = inputs.to(device)
        n_data = inputs.size(0)
        # 勾配を0に初期化する（逆伝播に備える）．
        optimizer.zero_grad()
        # 順伝播
        outputs = model(inputs)
        loss = criterion(outputs, inputs.view(n_data, -1))
        # 逆伝播
        loss.backward()
        # 重みの更新
        optimizer.step()

        sum_loss += loss.item() * n_data
        total += n_data

    return sum_loss / total

def test():
    global testloader, model, criterion, optimizer

    model.eval()
    sum_loss = 0
    total = 0

    with torch.no_grad():
        for (inputs, _) in testloader:
            inputs = inputs.to(device)
            n_data = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, inputs.view(n_data, -1))

            sum_loss += loss.item() * n_data
            total += n_data

    return sum_loss / total

def visualize():
    global n_epoch, train_loss, test_loss
    global LOSS_FILE

    epochs = np.arange(1, n_epoch+1)

    # loss の可視化
    plt.figure()

    plt.plot(epochs, train_loss, label="train", color='tab:blue')
    am = np.argmin(train_loss)
    plt.plot(epochs[am], train_loss[am], color='tab:blue', marker='x')
    plt.text(epochs[am], train_loss[am]+0.001, '%.3f' % train_loss[am], horizontalalignment="center", verticalalignment="bottom")

    plt.plot(epochs, test_loss, label="test", color='tab:orange')
    am = np.argmin(test_loss)
    plt.plot(epochs[am], test_loss[am], color='tab:orange', marker='x')
    plt.text(epochs[am], test_loss[am]-0.001, '%.3f' % test_loss[am], horizontalalignment="center", verticalalignment="top")

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.title('loss')
    plt.savefig(LOSS_FILE)

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
if not os.path.isdir('graph'):
    os.mkdir('graph')

t = datetime.datetime.now().strftime('%m%d-%H%M')
CKPT_FILE = './checkpoint/ckpt-%s.pth' % t
LOSS_FILE = './graph/loss-%s.png' % t

for epoch in range(1, n_epoch+1):
    train_loss += [train()]
    test_loss += [test()]
    print('epoch %2d | train_loss: %.3f, test_loss: %.3f'
        % (epoch, train_loss[-1], test_loss[-1]))

# checkpoint の保存
print('Saving..')
state = {
    'net': model.state_dict(),
    'acc': test_loss[-1],
    'epoch': epoch,
}
torch.save(state, CKPT_FILE)

visualize()