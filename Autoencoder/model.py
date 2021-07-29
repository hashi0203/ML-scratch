import torch.nn as nn

# 自己符号化器
class Autoencoder(nn.Module):
    in_dim = 784
    hid_dim = 80
    out_dim = in_dim

    def __init__(self):
        super(Autoencoder, self).__init__()

        fcs = []
        fcs.append(nn.Linear(self.in_dim, self.hid_dim))
        fcs.append(nn.ReLU())
        fcs.append(nn.Linear(self.hid_dim, self.out_dim))
        fcs.append(nn.Sigmoid())
        self.fcs = nn.Sequential(*fcs)

    def forward(self, x):
        x = x.view(-1, self.in_dim)
        return self.fcs(x)