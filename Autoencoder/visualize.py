import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from model import Autoencoder

import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Visualizer')
parser.add_argument('ckpt', metavar='FILE', type=str, help='ckpt to be used')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Autoencoder().to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model) # DataParallel を使って高速化
    cudnn.benchmark = True

# checkpoint の読み込み
checkpoint = torch.load(args.ckpt)
model.load_state_dict(checkpoint['net'])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

if not os.path.isdir('images'):
    os.mkdir('images')

def show_images(inputs, idx):
    global args, model
    outputs = model(inputs)

    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=False, figsize=(16, 8))

    for i, (in_img, out_img) in enumerate(zip(inputs, outputs)):
        axes[0, i].imshow(in_img.cpu().reshape(28, 28).detach().numpy(), cmap='gray')
        axes[1, i].imshow(out_img.cpu().reshape(28, 28).detach().numpy(), cmap='gray')

    fig.tight_layout()
    plt.savefig('./images/noise%d-%s.png' % (idx, args.ckpt[-13:-4]))

inputs, _ = iter(testloader).next()
inputs = inputs.to(device)

# ratio だけ元の画像を維持し，残りを白にする
ratio = 0.90

mask = torch.rand_like(inputs, dtype=torch.float)
mask = torch.ceil(mask - ratio)
show_images(mask + (1 - mask) * inputs, 0)

# std, maen のガウシアンノイズを載せる
std = 0.3
mean = 0

noise = torch.randn_like(inputs) * std + mean
show_images(inputs + noise, 1)






