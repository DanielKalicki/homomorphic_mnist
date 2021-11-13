import torch
import torch.optim as optim
import torch.nn.functional as F
from batchers.mnist_batcher import MnistBatch
from models.mnist_classifier import MnistClassifier
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import sys
from tqdm import tqdm
import numpy as np
import math
import random

config_idx = int(sys.argv[1])

config = {
    'batch_size': 32,
    'image_h': 28,
    'image_w': 28,
    'class_num': 10,
    'name': '_' + str(config_idx),
    'training': {
        'log': True,
        'lr': 1e-3,
        'lr_gamma': 0.99,
        'epochs': 1000
    }
}

config['training']['log'] = False

train_step = 0
test_step = 0

if config['training']['log']:
    now = datetime.now()
    writer = SummaryWriter(log_dir="./train/logs/"+config['name'])

class LabelSmoothingCrossEntropy(torch.nn.Module):
    # based on https://github.com/seominseok0429/label-smoothing-visualization-pytorch
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.2):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return torch.mean(loss)

def train(model, device, loader, optimizer, epoch, scheduler):
    loss = run(model, device, loader, optimizer, epoch, mode="train", scheduler=scheduler)
    return loss

def test(model, device, loader, optimizer, epoch, scheduler):
    loss = run(model, device, loader, optimizer, epoch, mode="test", scheduler=scheduler)
    return loss

def run(model, device, loader, optimizer, epoch, mode="train", scheduler=None):
    global test_step, train_step
    if mode == "test":
        model.eval()
    loss_pred = 0.0
    total = 0
    correct = 0

    start = time.time()
    pbar = tqdm(total=len(loader), dynamic_ncols=True)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    for batch_idx, (data, label) in enumerate(loader):
        if mode == 'train':
            train_step += 1
        elif mode == 'test':
            test_step += 1

        data, label = data.to(device), label.to(device)
        if mode == "train":
            optimizer.zero_grad()

        pred = model(data)
        loss = F.cross_entropy(pred, label, reduction='mean')

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

        _, predicted = torch.max(pred, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        if not math.isnan(float(loss)):
            loss_pred += loss.detach()
            if mode == "train":
                loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        pbar.update(1)
        pbar.set_description(str(round(correct/(float(total)+1e-6), 4)))

    pbar.close()
    end = time.time()
    loss_pred /= batch_idx + 1
    if math.isnan(float(loss)):
        print("nan")
        exit(0)
    print("")
    print('Epoch {}:'.format(epoch))
    print('\t\t' + mode + ' time: {:.2f}'.format((end - start)))
    if config['training']['log']:
        writer.add_scalar('loss_pred/'+mode, loss_pred, epoch)
        writer.add_scalar('acc/'+mode, correct/(float(total)+1e-6), epoch)
        writer.flush()
    return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = MnistClassifier(config)
print(model)
model.to(device)
start_epoch = 1

dataset_train = MnistBatch(config)
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=config['batch_size'],
    shuffle=True, num_workers=4)
dataset_test = MnistBatch(config, valid=True)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=config['batch_size'],
    shuffle=False, num_workers=0)

optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
lr_lambda = lambda epoch: config['training']['lr_gamma']**epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
test_loss = 1e6
for epoch in range(start_epoch, config['training']['epochs'] + start_epoch):
    train(model, device, data_loader_train, optimizer, epoch, None)
    current_test_loss = test(model, device, data_loader_test, None, epoch, None)
    dataset_train.on_epoch_end()
    dataset_test.on_epoch_end()
    print("mean="+str(test_loss))
    scheduler.step()

if config['training']['log']:
    writer.close()