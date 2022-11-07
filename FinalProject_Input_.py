import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import shutil
import zipfile
import glob
import os
import time
from torchsummary import summary


#FolderName
data_Folder_Front = 'C:\\Users\\sean6\\Pictures\\FreeVideoToJPGConverter\\'
data_Folder_Back = 'Label_1'
data_Folder = data_Folder_Front+data_Folder_Back
Label_Name = "LABEL1_"


print(f'the number of train set : {len(os.listdir(data_Folder))}')


i=0
file = os.listdir(data_Folder)
for name in file:
    picture = os.path.join(data_Folder,name)
    dst = Label_Name+str(i) + '.jpg'
    dst = os.path.join(data_Folder,dst)
    os.rename(picture,dst)
    i+=1
    
# https://hogni.tistory.com/35



