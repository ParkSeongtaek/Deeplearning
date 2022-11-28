# https://github.com/jo1jun/Vision_Transformer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import matplotlib.pyplot as plt
import os
import copy
import pickle
from PIL import Image
from urllib.request import urlretrieve
# tensorboard writer
from torch.utils.tensorboard import SummaryWriter

import typing
import io
import os

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from urllib.request import urlretrieve

from PIL import Image
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.long


# Hyperparameter
channel = 3
patch_size = 4

d_model = 128
n_layers = 6
nhead = 4
#n_layers = 12
#nhead = 8



ff_dim = 1024
dropout_ratio = 0.2
output_dim = 10

num_epochs = 150
learning_rate = 0.01
resize = 32

train_acc_list=[]
test_acc_list = []

title = "VIT_n_layers{}_nhead{}_resize{}".format(n_layers,nhead,resize)


# tensor 연산을 간편하게 하기 위해 einops moudule import.
# reference & tutorial : http://einops.rocks/pytorch-examples.html
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# WarmupCosineSchedule 이용
from pytorch_transformers.optimization import WarmupCosineSchedule

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout_ratio):
        super().__init__()

        self.d_model = d_model # d_model : embed dimension
        self.nhead = nhead # nhead : head 수
        self.head_dim = d_model // nhead # head_dim : head 마다의 dimension

        # 위 figure 에서 Q, K, V 가 head 수 만큼의 Linear 를 거쳐 Scaled Dot-Product Attention에 들어가는데
        # 실제 구현에서는 Q, K, V 각각이 하나의 Linear 를 거쳐 나온 것을 쪼개서 Scaled Dot-Product Attention에 들어가게 함.
        self.qLinear = nn.Linear(d_model, d_model) # query lineqr
        self.kLinear = nn.Linear(d_model, d_model) # key linear
        self.vLinear = nn.Linear(d_model, d_model) # value linear

        self.oLinear = nn.Linear(d_model, d_model) # output linear

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, query, key, value): # 기존의 Transformer 와 달리 mask 없음.

        batch_size = query.shape[0]
        query_len = query.shape[1]
        value_len = key_len = key.shape[1]

        # query: [batch_size, query_len, d_model]
        # key: [batch_size, key_len, d_model]
        # value: [batch_size, value_len, d_model]
 
        Q = self.qLinear(query)
        K = self.kLinear(key)
        V = self.vLinear(value)

        # Q: [batch_size, query_len, d_model]
        # K: [batch_size, key_len, d_model]
        # V: [batch_size, value_len, d_model]

        # d_model = nhead * head_dim
        # head 수로 distribute
        Q = rearrange(Q, 'b l (h d) -> b h l d', h=self.nhead)
        K = rearrange(K, 'b l (h d) -> b h l d', h=self.nhead)
        V = rearrange(V, 'b l (h d) -> b h l d', h=self.nhead)

        # Q: [batch_size, nhead, query_len, head_dim]
        # K: [batch_size, nhead, key_len, head_dim]
        # V: [batch_size, nhead, value_len, head_dim]

        #=========Scaled Dot-Product Attention=========

        # attention weight 계산
        weight = torch.matmul(Q, rearrange(K, 'b h l d -> b h d l')) / np.sqrt(self.head_dim)

        # weight: [batch_size, nhead, query_len, key_len]

        attention = torch.softmax(weight, dim=-1)

        # attention: [batch_size, nhead, query_len, key_len]

        # context vector 계산
        c = torch.matmul(self.dropout(attention), V)

        # c: [batch_size, nhead, query_len, head_dim]

        #=========Scaled Dot-Product Attention=========

        # reshape & concat
        c = rearrange(c, 'b h l d -> b l (h d)')

        # c: [batch_size, query_len, d_model]

        output = self.oLinear(c)

        # output: [batch_size, query_len, d_model]

        return output, attention
    
class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, d_model, ff_dim, dropout_ratio):
        super().__init__()
        # ff_dim : FFN 에서 hidden layer node 수
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):

        # x: [batch_size, seq_len, d_model]

        # The MLP contains two layers with a GELU non-linearity
        x = self.dropout(nn.functional.gelu(self.linear1(x)))

        # x: [batch_size, seq_len, ff_dim]

        x = self.linear2(x)

        # x: [batch_size, seq_len, d_model]

        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, ff_dim, dropout_ratio):
        super().__init__()

        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)
        self.multiHeadAttentionLayer = MultiHeadAttentionLayer(d_model, nhead, dropout_ratio)
        self.positionWiseFeedForward = PositionWiseFeedForwardLayer(d_model, ff_dim, dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, src):

        # src: [batch_size, src_len, d_model] # encoder layer 에 입력되는 값

        # Layernorm (LN) is applied before every block
        _src = self.layerNorm1(src)

        # self attention 이므로 query, key, value 전부 동일
        _src, attention = self.multiHeadAttentionLayer(_src, _src, _src)

        # residual connections after every block

        # Dropout, when used, is applied after every dense layer except for the the qkv-projections
        # and directly after adding positional- to patch embeddings. 
        src = src + self.dropout(_src)

        _src = self.layerNorm2(src)

        # src: [batch_size, src_len, d_model]

        _src = self.positionWiseFeedForward(_src)

        src = src + self.dropout(_src)

        # src: [batch_size, src_len, d_model]

        return src, attention
    
class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, nhead, ff_dim, dropout_ratio):
        super().__init__()
        # n_layers : encoder layer 반복 수
        # Encoder Layer 을 n_layers 만큼 반복.
        self.layers = nn.ModuleList([EncoderLayer(d_model, nhead, ff_dim, dropout_ratio) for _ in range(n_layers)])

    def forward(self, src):

        # src: [batch_size, src_len, d_model]

        for layer in self.layers:
            src, attention = layer(src)

        # src: [batch_size, src_len, d_model]

        return src, attention
    
class ImageEmbedding(nn.Module):
  def __init__(self, channel, patch_size, D):
    super().__init__()
    # image 를 patch 단위로 나눈 뒤 [batch_size, patch 개수, patch_size * patch_size * channel] 로 shape 만들기
    self.rearrange = Rearrange('b c (num_w p1) (num_h p2) -> b (num_w num_h) (p1 p2 c) ', p1=patch_size, p2=patch_size)
    # 임베딩 작업 [batch_size, patch 개수, patch_size * patch_size * channel] -> [batch_size, patch 개수, embed_dim]
    self.linear = nn.Linear(channel*patch_size*patch_size, D)
    # learnable cls_token (only for classification)
    self.cls_token = nn.Parameter(torch.randn(1, 1, D))

  def forward(self, image):

    b, c, w, h = image.shape

    # image: [batch_size, channel, width, height]

    # num_w = w / p1 # num_w : width 가 패치로 나누어진 수
    # num_h = h / p2 # num_h : height 가 패치로 나누어진 수
    # n = num_w * num_h = (w*h) / (p1*p2)
    flatten_patches = self.rearrange(image)

    # flatten_patches : [batch_size, patch 개수, patch_size * patch_size * channel]

    embedded_patches = self.linear(flatten_patches)

    # embedded_patches : [batch_size, patch 개수, embed_dim]
    
    # Similar to BERT’s [class] token, we prepend a learnable embedding to the sequence of embedded patches
    # classification 의 경우 맨 앞에 learnable cls_token 추가.
    cls_tokens = repeat(self.cls_token, 'b n d -> (b repeat) n d', repeat=b)
    embedded_patches = torch.cat((cls_tokens, embedded_patches), dim=1)

    # embedded_patches : [batch_size, 1 + patch 개수, embed_dim]

    return embedded_patches


class TokPosEmbedding(nn.Module):
  def __init__(self, c, p, d_model, dropout_ratio):
    super().__init__()
    self.tokEmbedding = ImageEmbedding(c, p, d_model)
    self.posEmbedding = nn.Embedding(100, d_model) # cos & sin positional encoding 대신, learnable positional embedding 으로 구현. (max patch 수 100)
                                                   # max length : 입력 sequence 의 최대 길이.
    self.d_model = d_model
    self.dropout = nn.Dropout(dropout_ratio)

  def forward(self, src):

    # src: [batch_size, width, height, channel]

    src = self.tokEmbedding(src)

    # src: [batch_size, src_len, d_model]

    batch_size = src.shape[0]
    src_len = src.shape[1]

    # 0 ~ src_len 등차 수열값 을 bactch 한개와 동일한 shape으로 생성 (positional embedding)
    pos = torch.arange(0, src_len, dtype=dtype) # pos: [src_len]
    pos = repeat(pos, 'l -> b l', b=batch_size).to(device) # pos: [batch_size, src_len]

    src = self.dropout(src * float(np.sqrt(self.d_model)) + self.posEmbedding(pos))
    # Dropout, when used, is applied after every dense layer except for the the qkv-projections and directly after adding positional- to patch embeddings

    # src: [batch_size, src_len, d_model]

    return src

class VisionTransformer(nn.Module):
    def __init__(self, c, p, d_model, n_layers, nhead, ff_dim, dropout_ratio, output_dim):
        super().__init__()

        self.encEmbedding = TokPosEmbedding(c, p, d_model, dropout_ratio)
        self.encoder = Encoder(d_model, n_layers, nhead, ff_dim, dropout_ratio)
        self.layerNorm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, output_dim) 

    def forward(self, src):
        
        # image: [batch_size, channel, width, height]

        src = self.encEmbedding(src)

        # src: [batch_size, src_len, d_model]

        enc_src, attention = self.encoder(src)

        # enc_src: [batch_size, src_len, d_model] # encoder의 출력값.

        # classification head

        enc_src = enc_src[:,0,:] # cls tokken

        # enc_src: [batch_size, d_model]

        enc_src = self.layerNorm(enc_src)

        output = self.linear(enc_src)

        # output: [batch_size, output_dim]

        return output, attention
    
import torchvision.datasets as dset
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import sampler

# here we set up a transform to preprocess the data by subtracting the mean RGB value 
# and dividing by the standard deviation of each RGB value; we've hardcoded the mean and std.

# 박성택 : 데이터 통일성을 위해 사진의 Normalize부분을 뺴고 T.Resize((32, 32)) 로 바꿔줬습니다. (CNN DNN 에 사용한 데이터가 3 32 32 입니다.)

transform = T.Compose([
                T.ToTensor(),
                T.Resize((resize, resize)),
                #T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

cifar10_train = dset.CIFAR10('dataset', train=True, download=True,
                             transform=transform)

cifar10_val = dset.CIFAR10('dataset', train=True, download=True,
                           transform=transform)

cifar10_test = dset.CIFAR10('dataset', train=False, download=True, 
                            transform=transform)

NUM_TRAIN = 49000

dataloaders = {}
dataloaders['train'] = DataLoader(cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
dataloaders['val'] = DataLoader(cifar10_val, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
dataloaders['test'] = DataLoader(cifar10_test, batch_size=64)



def trainer(model_name, model, criterion, optimizer, scheduler, num_epochs):

    model.to(device)
    writer = SummaryWriter(f'runs/{model_name}')
    best_model_wts = copy.deepcopy(model.state_dict())
    global_step, best_acc = 0, 0.0
    running_loss, running_acc = {}, {}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss[phase], running_acc[phase] = 0.0, 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, _ = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss[phase] += loss.item() * inputs.shape[0]
                running_acc[phase] += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()
            
            running_loss[phase] = running_loss[phase] / (len(dataloaders[phase]) * dataloaders[phase].batch_size)
            running_acc[phase] = running_acc[phase].double() / (len(dataloaders[phase]) * dataloaders[phase].batch_size)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, running_loss[phase], running_acc[phase]))
            
            if phase == 'train':
                train_acc_list.append(float('{:.4f}'.format(running_acc[phase])))
                print("running_acc[phase]" + '{:.4f}'.format(running_acc[phase]))
            else: 
                test_acc_list.append(float('{:.4f}'.format(running_acc[phase])))
                print("running_acc[phase]" + '{:.4f}'.format(running_acc[phase]))
            
		   
           

            
            # deep copy the model
            if phase == 'val' and running_acc[phase] > best_acc:
                best_acc = running_acc[phase]
                best_model_wts = copy.deepcopy(model.state_dict())

        writer.add_scalars(f'{model_name}/loss', {'train' : running_loss['train'], 'val' : running_loss['val']}, global_step)
        writer.add_scalars(f'{model_name}/acc', {'train' : running_acc['train'], 'val' : running_acc['val']}, global_step)

        

        writer.add_scalar(f'{model_name}/lr', scheduler.get_last_lr()[0], global_step)

        global_step += 1

        print()

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    torch.save(model.state_dict(), f'{model_name}.pt')
    print('model saved')

    writer.close()

    return model

def checker(loader, model):

    model.eval()
    with torch.no_grad():
      correct_num = 0
      for iters, (batch_x, batch_t) in enumerate(loader):

        batch_x = batch_x.to(device)
        batch_t = batch_t.to(device)

        predict, _ = model(batch_x)
        _, predict = predict.max(1)

        correct_num += (predict == batch_t).sum()
        
    return correct_num

model = VisionTransformer(channel, patch_size, d_model, n_layers, nhead, ff_dim, dropout_ratio, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)
scheduler = WarmupCosineSchedule(optimizer, num_epochs // 5, num_epochs)
model = trainer('ViT_CIFAR10', model, criterion, optimizer, scheduler, num_epochs)

correct_num = checker(dataloaders['train'], model)
acc = float(correct_num) / NUM_TRAIN
print('train accuracy %.3f%%' % (acc * 100))

correct_num = checker(dataloaders['val'], model)
acc = float(correct_num) / (50000 - NUM_TRAIN)
print('val accuracy %.3f%%' % (acc * 100))

correct_num = checker(dataloaders['test'], model)
acc = float(correct_num) / len(cifar10_test)
print('test accuracy %.3f%%' % (acc * 100))

#저장 출력
train_acc_list_title = 'train_acc_'+ title +'.pickle'
test_acc_list_title = 'test_acc_' + title + '.pickle'

# save
with open(train_acc_list_title , 'wb') as fw:
    pickle.dump(train_acc_list, fw)
    

# save
with open(test_acc_list_title, 'wb') as fw:
    pickle.dump(test_acc_list, fw)
    

plt.plot(train_acc_list, 'r--', test_acc_list, 'bs')	
plt.show()

