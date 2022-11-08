import os
import numpy as np
from PIL import Image
import pickle
#변환할 이미지 목록 불러오기

ReShapeSize = 32


data = []
### 피클 파일 불러오기 ###
with open("ForDebug_DNN_32.pickle","rb") as fr:
    data = pickle.load(fr)

print(data)


#jpg to array
#https://thinking-developer.tistory.com/62

#reshape
#https://ponyozzang.tistory.com/600

