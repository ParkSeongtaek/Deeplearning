import os
import numpy as np
from PIL import Image
import pickle
import random
#변환할 이미지 목록 불러오기

ReShapeSize = 32


image_path = ["" for i in range (4)]
image_path[0] = 'C:/Users/sean6/Pictures/FreeVideoToJPGConverter/Label_0/'
image_path[1] = 'C:/Users/sean6/Pictures/FreeVideoToJPGConverter/Label_1/'
image_path[2] = 'C:/Users/sean6/Pictures/FreeVideoToJPGConverter/Label_2/'
image_path[3] = 'C:/Users/sean6/Pictures/FreeVideoToJPGConverter/Label_3/'

img_list = ["" for i in range (4)]
for label in range(4):
    img_list[label] = os.listdir(image_path[label]) #디렉토리 내 모든 파일 불러오기


for label in range(4):
    random.shuffle(img_list[label])
    

for label in range(4):
    for idx in range(10):
        print(img_list[label][idx])
    print("********************************************************")
    

img_list_jpg = ["" for i in range (4)]
for label in range(4):
    #### test 를 위한 data숫자 통제                       *여기*  이후 원본 데이터를 위해 삭제 
    img_list_jpg[label] = [img for img in img_list[label][:400] if img.endswith(".jpg")] #지정된 확장자만 필터링


img_list_np = []
for label in range(4):
    for i in img_list_jpg[label]:
        label_table = [0 for i in range(4)]
        label_table[label] = 1 
        img = Image.open(image_path[label] + i)        #Image형태로 열기
        img = img.resize((ReShapeSize, ReShapeSize))            #원하는 Size로 reshape
        img_array = np.array(img)               #np형태로 변환

        img_array = img_array.ravel()             #DNN을 위한 Data 1차원으로 변환
        #print(img_array.shape)
        img_tuple = (img_array,label_table)
        
        img_list_np.append(img_tuple)           
        #print(i, " 추가 완료 - 구조:", img_array.shape) # 불러온 이미지의 차원 확인 (세로X가로X색)
        #print(img_array.T.shape) #축변경 (색X가로X세로)

img_list_np = np.asarray(img_list_np)
np.random.shuffle(img_list_np)

print (img_list_np.shape)
print (img_list_np)

## Save pickle
with open("ForDebug_DNN_32.pickle","wb") as fw:
    pickle.dump(img_list_np, fw)

 


#jpg to array
#https://thinking-developer.tistory.com/62

#reshape
#https://ponyozzang.tistory.com/600

