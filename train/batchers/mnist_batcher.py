import numpy as np
import torch
from torch.utils.data import Dataset
# import torchvision
import os
import codecs
# import matplotlib.pyplot as plt

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
        # torchvision.datasets.MNIST('./train/datasets', train=True, download=True)
        # torchvision.datasets.MNIST('./train/datasets', train=False, download=True)
        self.mnist_train_images = read_image_file(datasets_dir+'MNIST/raw/train-images-idx3-ubyte')
        self.mnist_train_labels = read_label_file(datasets_dir+'MNIST/raw/train-labels-idx1-ubyte')

        # to_pil_image = torchvision.transforms.ToPILImage(mode=None)
        # center_crop = torchvision.transforms.CenterCrop(25)
        # resize = torchvision.transforms.Resize(self.config['image_h'], interpolation=5)
        # pil_to_tensor = torchvision.transforms.ToTensor()

        # train_batch = []
        # train_labels = []
        # for b_idx in range(self.mnist_train_images.shape[0]):
        #     if self.mnist_train_labels[b_idx] in self.config['classes']:
        #         print('----')
        #         data = self.mnist_train_images[b_idx]
        #         for col in data:
        #             for row in col:
        #                 print(round(float(row),2), end='\t')
        #             print('')
        #         pil_image = to_pil_image(self.mnist_train_images[b_idx])
        #         pil_image = center_crop(pil_image)
        #         pil_image = resize(pil_image)
        #         image = pil_to_tensor(pil_image).squeeze_(0)
        #         print(self.mnist_train_labels[b_idx])
        #         for col in image:
        #             for row in col:
        #                 print(round(float(row),4), end='\t')
        #             print('')
        #         train_batch.append(image)
        #         if len(train_batch) > 100:
        #             quit()
        #         train_labels.append(self.config['classes'].index(self.mnist_train_labels[b_idx]))
        # self.mnist_train_images = train_batch
        # self.mnist_train_labels = train_labels

    def on_epoch_end(self):
        pass

    def __len__(self):
        return len(self.mnist_train_images)

    def __getitem__(self, idx):
        data = self.mnist_train_images[idx]
        target = self.mnist_train_labels[idx]
        data = (data).type(torch.FloatTensor)
        data = data / 256.0

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
