import os
import numpy as np
from PIL import Image
import pickle
import random
import natsort ## 숫자 정렬용 라이브러리
from einops import rearrange, repeat, reduce
#변환할 이미지 목록 불러오기

'''
LABELS = ['frog', 'truck', 'deer', 'automobile', 'bird', 'horse', 'ship',
          'cat', 'dog', 'airplane']

image_path = 'C:/Users/sean6/Desktop/DeepLearning_main/prototype/cifar-10/train/train/'
img_list = os.listdir(image_path) #디렉토리 내 모든 파일 불러오기
img_list = natsort.natsorted(img_list)  #순서대로 소팅

loadedData = np.loadtxt(open("C:\\Users\\sean6\\Desktop\\DeepLearning_main\\prototype\\cifar-10\\trainLabelsMYMY.csv","rb"),delimiter=",", dtype=np.str)

label = loadedData[:,1:]        
label = label.ravel()           #label 순서대로  
img_list
img_list_png = [img for img in img_list if img.endswith(".png")] #지정된 확장자만 필터링


img_list_np = []
img_list_np = []
idx = 0
for i in img_list_png:
    #label_table = [0 for i in range(10)]
    #label_table[LABELS.index(label[idx])] = 1 
    label_table = LABELS.index(label[idx]) 
    
    #print(label_table)
    
    img = Image.open(image_path + i)        #Image형태로 열기
    #print(type(img))
    #img.show()
    img_array = np.array(img)               #np형태로 변환
    
    #img_array = img_array.ravel()             #DNN을 위한 Data 1차원으로 변환
    
    img_array = rearrange(img_array, 'r h C -> C r h')
    img_array = np.array(img_array)/255
    #print(img_array.shape)
    img_tuple = (img_array,label_table)
    img_list_np.append(img_tuple)           
    #print(i, " 추가 완료 - 구조:", img_array.shape) # 불러온 이미지의 차원 확인 (세로X가로X색)
    #print(img_array.T.shape) #축변경 (색X가로X세로)
    idx += 1
    if idx%100 == 0:
        print(idx/100)
print (img_list_np.__len__())



with open("CIRFAR_10_CNN_32.pickle","wb") as fw:
    pickle.dump(img_list_np, fw)

'''
with open("CIRFAR_10_CNN_32.pickle","rb") as fr:
    img_list_np = pickle.load(fr)


print(img_list_np.__len__())
for idx in range(1):
    print(img_list_np[idx][0])
    print(img_list_np[idx][1])




#jpg to array
#https://thinking-developer.tistory.com/62

#reshape
#https://ponyozzang.tistory.com/600

'''
import numpy
#my_matrix = numpy.loadtxt(open("C:\\Users\\sean6\\Desktop\\DeepLearning_main\\prototype\\cifar-10\\trainLabelsMYMY.csv","rb"),delimiter=",",skiprows=0)
idx = loadedData[:,1:2]

print(idx)

idx = np.squeeze(idx, axis=1)

print(idx)

'''