import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random
# 하이퍼파라미터
title1 = 'test_acc_Resnet.pickle'
title2 = 'train_acc_Resnet.pickle'

title3 = 'test_acc_VIT_n_layers6_nhead4_resize32.pickle'
title4 = 'train_acc_VIT_n_layers6_nhead4_resize32.pickle'


data1 = []
data2 = []
data3 = []
data4 = []

with open(title1,"rb") as fr:
    data1 = pickle.load(fr)
with open(title2,"rb") as fr:
    data2 = pickle.load(fr)
with open(title3,"rb") as fr:
    data3 = pickle.load(fr)
with open(title4,"rb") as fr:
    data4 = pickle.load(fr)

plt.title("Blue: Resnet / Red: VIT") #그래프에 제목 넣기

plt.plot(data1, 'bs',  data2, 'b--', data3, 'rs', data4, 'r--')#, data5, 'gs', data6, 'g--')#,data7,'ks' , data8,'k--')	

plt.show()
