import pickle
import numpy as np
import pandas as pd

from two_layer_net import TwoLayerNet
# 하이퍼파라미터
iters_num = 5000 # 반복 횟수를 적절히 설정한다.
train_size = 1000  # 640 장 중 600 장을 train 으로 사용
data_num = 1400
batch_size = 40  # 미니배치 크기 -> 40 장
learning_rate = 0.1
RGB =3
PIXEL = 32
HIDDEN_SIZE =50
OUTPUT_SIZE = 4

pickle_name = "ForDebug_DNN_{}.pickle".format(PIXEL)

with open(pickle_name,"rb") as fr:
    data = pickle.load(fr)

# 무작위로 섞는 부분
data = pd.DataFrame(data)
data = data.sample(frac=1, replace=False)
# 다시 numpy 로 만들어준다.
data = data.to_numpy()
# label 과 data를 분리
label = [x[1] for x in data]
data = [x[0] for x in data]

data = (np.array(data, dtype='float64')) / 255
label = np.array(label)
# print(data)
# print(data.shape)
# print(label.shape)
x_train = data[:train_size]
t_train = label[:train_size]
x_test = data[train_size:]
t_test = label[train_size:]

network = TwoLayerNet(input_size = RGB * PIXEL * PIXEL, hidden_size = HIDDEN_SIZE, output_size = OUTPUT_SIZE)


train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
	# 미니배치 획득
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	# 기울기 계산
	# grad = network.numerical_gradient(x_batch, t_batch)
	grad = network.gradient(x_batch, t_batch)

	# 매개변수 갱신
	for key in ('W1', 'b1', 'W2', 'b2'):
		network.params[key] -= learning_rate * grad[key]

	# 학습 경과 기록
	loss = network.loss(x_batch, t_batch)
	train_loss_list.append(loss)

	# 1에폭당 정확도 계산
	if i % iter_per_epoch == 0:
		train_acc = network.accuracy(x_train, t_train)
		test_acc = network.accuracy(x_test, t_test)
		train_acc_list.append(train_acc)
		test_acc_list.append(test_acc)
		print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
