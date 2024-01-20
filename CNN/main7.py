
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

PATH_train = "data/Im"
TRAIN = Path(PATH_train)
PATH_train2 = "data/Im2"
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
train_size = int(0.7 * len(train_data))
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


def train(model, n_epochs, train_loader, valid_loader, optimizer, criterion):
    train_acc_his, valid_acc_his = [], []
    train_losses_his, valid_losses_his = [], []
    for epoch in range(1, n_epochs + 1):
        # keep track of training and validation loss
        train_loss, valid_loss = 0.0, 0.0
        train_losses, valid_losses = [], []
        train_correct, val_correct, train_total, val_total = 0, 0, 0, 0
        train_pred, train_target = torch.zeros(8, 1), torch.zeros(8, 1)
        val_pred, val_target = torch.zeros(8, 1), torch.zeros(8, 1)
        count = 0
        count2 = 0
        print('running epoch: {}'.format(epoch))
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in tqdm(train_loader):
            # move tensors to GPU if CUDA is available
           # if train_on_gpu:
                #data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # calculate accuracy
            pred = output.data.max(dim=1, keepdim=True)[1]
            train_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            train_total += data.size(0)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_losses.append(loss.item() * data.size(0))
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            if count == 0:
                train_pred = pred
                train_target = target.data.view_as(pred)
                count = count + 1
            else:
                train_pred = torch.cat((train_pred, pred), 0)
                train_target = torch.cat((train_target, target.data.view_as(pred)), 0)
        train_pred = train_pred.cpu().view(-1).numpy().tolist()
        train_target = train_target.cpu().view(-1).numpy().tolist()
        ######################
        # validate the model #
        ######################
        model.eval()
        for data, target in tqdm(valid_loader):
            # move tensors to GPU if CUDA is available
            #if train_on_gpu:
               # data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # calculate accuracy
            pred = output.data.max(dim=1, keepdim=True)[1]
            val_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            val_total += data.size(0)
            valid_losses.append(loss.item() * data.size(0))
            if count2 == 0:
                val_pred = pred
                val_target = target.data.view_as(pred)
                count2 = count + 1
            else:
                val_pred = torch.cat((val_pred, pred), 0)
                val_target = torch.cat((val_target, target.data.view_as(pred)), 0)
        val_pred = val_pred.cpu().view(-1).numpy().tolist()
        val_target = val_target.cpu().view(-1).numpy().tolist()

        # calculate average losses
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        # calculate average accuracy
        train_acc = train_correct / train_total
        valid_acc = val_correct / val_total

        train_acc_his.append(train_acc)
        valid_acc_his.append(valid_acc)
        train_losses_his.append(train_loss)
        valid_losses_his.append(valid_loss)
        # print training/validation statistics
        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, valid_loss))
        print('\tTraining Accuracy: {:.6f} \tValidation Accuracy: {:.6f}'.format(train_acc, valid_acc))





    #torch.save(model, 'data/model5.pt')
    return train_acc_his, valid_acc_his, train_losses_his, valid_losses_his, model



model1 = CNN_Model()
#model = torch.load('data/model.pt')


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

#test_set = ImgDataset(train_data)
#test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
#print(test_loader)
#print(test_loader.dataset)
#prediction = []
#with torch.no_grad():
   # for i, data in enumerate(test_loader):
        #print(data)
       # data = np.array(data).astype(np.float32)  # list转numpy.array
       # data = torch.from_numpy(data)  # array2tensor
       # test_pred = model(data)
       # print(test_pred)
      #  test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
       # print(test_label)
       # for y in test_label:
       #     prediction.append(y)


#model.eval()
#prediction = []
#with torch.no_grad():
    #for i, data in enumerate(TRAIN):
      #  data = np.array(i).astype(np.float32)
      #  data = torch.from_numpy(data)
       # data = data.cuda()

        #test_pred = model(data)
        #test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        #for y in test_label:
         #   prediction.append(y)

#test_loader = "data/dogcat/"

#def test(model, device, test_loader):
  #  model.eval()
   # test_loss = 0
   # correct = 0
   # with torch.no_grad():
   #    for data, target in test_loader:
    #        data, target = data.to('cuda'), target.to('cuda')
    #        output = model(data)
            #test_loss += f.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
     #       pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
     #       print(pred, target)
     #       correct += pred.eq(target.view_as(pred)).sum().item()
#
    #test_loss /= len(test_loader.dataset)

   # print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     correct, len(test_loader.dataset),
     #   100. * correct / len(test_loader.dataset)))

#test(model, 'cuda', test_loader)

#model1.eval()
#prediction = []
#with torch.no_grad():
   # for data, target in tqdm(test_loader):
    #    test_pred = model1(data)
     #   test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
     #   for y in test_label:
     #       prediction.append(y)
#print(prediction)



n_epochs = 10
optimizer1 = torch.optim.Adam(model1.parameters(), lr=LR)
criterion = CrossEntropyLoss()
train_acc_his, valid_acc_his, train_losses_his, valid_losses_his, model1 = train(model1, n_epochs, train_loader,
                                                                                 valid_loader, optimizer1, criterion)
plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(train_losses_his, 'bo', label = 'training loss')
plt.plot(valid_losses_his, 'r', label = 'test loss')
plt.title("CNN Loss")
plt.legend(loc='upper left')
plt.subplot(222)
plt.plot(train_acc_his, 'bo', label = 'training accuracy')
plt.plot(valid_acc_his, 'r', label = 'test accuracy')
plt.title("CNN Accuracy")
plt.legend(loc='upper left')
plt.show()



model = torch.load('data/model5.pt')
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
            #   count += 1
            for y in test_label:
                prediction.append(y)
print(test_correct)
print(float(test_correct / 10.0))
print(prediction)