import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MnistClassifier(nn.Module):
    def __init__(self, config):
        super(MnistClassifier, self).__init__()
        self.config = config
        self.h = self.config['image_h']
        self.w = self.config['image_w']
        self.out_dim = self.config['class_num']
        self.fc1 = nn.Linear(self.h*self.w, 16)
        # self.act1 = F.leaky_relu
        self.fc2 = nn.Linear(16, self.out_dim)
        # self.act2 = F.relu
        # self.fc3 = nn.Linear(16, self.out_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x_hidd = x
        # x = self.act1(x)
        x = x**2
        x = self.fc2(x)
        # x = self.act2(x)
        # x = self.fc3(x)
        return x, x_hidd