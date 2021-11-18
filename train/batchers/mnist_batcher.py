import numpy as np
import torch
from torch.utils.data import Dataset
import os
import codecs
import random

# based on https://pytorch.org/docs/1.1.0/_modules/torchvision/datasets/mnist.html
def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)

def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()

class MnistBatch():
    def __init__(self, config, valid=False):
        self.config = config
        datasets_dir = './train/datasets/'
        self.valid = valid
        self.mnist_train_images = read_image_file(datasets_dir+'MNIST/raw/train-images-idx3-ubyte')
        self.mnist_train_labels = read_label_file(datasets_dir+'MNIST/raw/train-labels-idx1-ubyte')

        self.mnist_valid_images = self.mnist_train_images[0:5000]
        self.mnist_valid_labels = self.mnist_train_labels[0:5000]
        self.mnist_train_images = self.mnist_train_images[5000:]
        self.mnist_train_labels = self.mnist_train_labels[5000:]

    def on_epoch_end(self):
        pass

    def __len__(self):
        if not self.valid:
            return len(self.mnist_train_images)
        else:
            return len(self.mnist_valid_images)

    def __getitem__(self, idx):
        if not self.valid:
            data = self.mnist_train_images[idx]
            target = self.mnist_train_labels[idx]
        else:
            data = self.mnist_valid_images[idx]
            target = self.mnist_valid_labels[idx]


        data = (data).type(torch.FloatTensor)
        data = data / 256.0

        for i in range(0, data.shape[0]):
            for ii in range(0, data.shape[1]):
                if float(data[i,ii].item()) != 0.0:
                    rnd = random.random()
                    if rnd > data[i,ii]:
                        data[i,ii] = 0.0
                    else:
                        data[i,ii] = 1.0
        #         print(int(data[i,ii].item()), end='')
        #     print("")
        # print(target)

        return data, target

config = {
    'batch_size': 1,
    'image_h': 2,
    'image_w': 2,
    'network_type': 'quantum',
    'classic': {
        'linear': False
    },
    'quantum': {
        'layers': 1
    },
    'classes': [0, 4],
    'name': ''
}

test = MnistBatch(config)
# print(test.__getitem__(1))
