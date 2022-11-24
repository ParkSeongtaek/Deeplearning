import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import os
import torchvision.models as models
import multiprocessing as mp
from multiprocessing import freeze_support # freeze_support 함수 임포트
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simple Learning Rate Scheduler
def lr_scheduler(optimizer, epoch):
    lr = learning_rate
    if epoch >= 50:
        lr /= 10
    if epoch >= 100:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Xavier         
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)
device = 'cuda'
model = ResNet50()
# ResNet18, ResNet34, ResNet50, ResNet101, ResNet152 중에 택일하여 사용
model.apply(init_weights)
model = model.to(device)
learning_rate = 0.001
num_epoch = 150
model_name = 'model.pth'

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss = 0
valid_loss = 0
correct = 0
total_cnt = 0
best_acc = 0
train_acc = []
test_acc = []
def Train() :
    global  train_loss
    global  valid_loss
    global  correct 
    global  total_cnt 
    global  best_acc 

    for epoch in range(num_epoch):
        print(f"====== { epoch+1} epoch of { num_epoch } ======")
        model.train()
        lr_scheduler(optimizer, epoch)
        train_loss = 0
        valid_loss = 0
        correct = 0
        total_cnt = 0
        # Train Phase
        for step, batch in enumerate(train_loader):
            #  input and target
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            logits = model(batch[0])
            loss = loss_fn(logits, batch[1])
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predict = logits.max(1)

            total_cnt += batch[1].size(0)
            correct +=  predict.eq(batch[1]).sum().item()

            if step % 100 == 0 and step != 0:
                print(f"\n====== { step } Step of { len(train_loader) } ======")
                print(f"Train Acc : { correct / total_cnt }")

                train_acc.append( correct / total_cnt)
		        
		

                print(f"Train Loss : { loss.item() / batch[1].size(0) }")

        correct = 0
        total_cnt = 0

    # Test Phase
        with torch.no_grad():
            model.eval()
            for step, batch in enumerate(test_loader):
                # input and target
                batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
                total_cnt += batch[1].size(0)
                logits = model(batch[0])
                valid_loss += loss_fn(logits, batch[1])
                _, predict = logits.max(1)
                correct += predict.eq(batch[1]).sum().item()
            valid_acc = correct / total_cnt
            print(f"\nValid Acc : { valid_acc }")    
            test_acc.append(valid_acc)
            print(f"Valid Loss : { valid_loss / total_cnt }")

            if(valid_acc > best_acc):
                best_acc = valid_acc
                torch.save(model, model_name)
                print("Model Saved!")


train_acc_list_title = 'train_acc_Resnet.pickle'
test_acc_list_title = 'test_acc_Resnet.pickle'
if __name__ == '__main__':   # 중복 방지를 위한 사용
    freeze_support()         # 윈도우에서 파이썬이 자원을 효율적으로 사용하게 만들어준다.

    Train()

    # save
    with open(train_acc_list_title , 'wb') as fw:
        pickle.dump(train_acc, fw)
    # save
    with open(test_acc_list_title, 'wb') as fw:
        pickle.dump(test_acc, fw)
    
    plt.plot(train_acc, 'r--', test_acc, 'bs')	
    plt.show()