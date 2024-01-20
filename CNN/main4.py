
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU, CrossEntropyLoss, Conv2d, MaxPool2d, Module
from torch.optim import Adam
import pandas as pd
import os
from os import listdir
from tqdm import tqdm
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
PATH_train = "data/dogcat"
TRAIN = Path(PATH_train)
PATH_train2 = "data/dogcat2"
TRAIN2 = Path(PATH_train2)

# Batch：每批丟入多少張圖片
batch_size = 8
# Learning Rate：學習率
LR = 0.0001
transforms = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

train_data = datasets.ImageFolder(TRAIN, transform=transforms)
test_data = datasets.ImageFolder(TRAIN2, transform=transforms)

print(train_data.class_to_idx)
# 切分70%當作訓練集、30%當作驗證集
train_size = int(0.8 * len(train_data))
valid_size = len(train_data) - train_size

train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])
# Dataloader可以用Batch的方式訓練
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

class CNN_Model(nn.Module):
    #列出需要哪些層
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(3,224,224)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)



model = torch.load('data/model3.pt')
#prediction = []
#with torch.no_grad():
   # for data, target in tqdm(test_loader):
    #    test_pred = model(data)
     #   test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
      #  for y in test_label:
      #      prediction.append(y)
#print(prediction)
model.eval()
prediction = []
true = []
test_correct = 0
for data, target in tqdm(test_loader):
    test_pred = model(data)
    pred = test_pred.data.max(dim=1, keepdim=True)[1]
    test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
    print(target.data)
    test_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
    # if test_pred == test_label:

    for y in test_label:
        prediction.append(y)
print(test_correct)
print(float(test_correct /72))
print(prediction)