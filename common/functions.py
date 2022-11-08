import numpy as np


def identity_function(x):
	return x


def step_function(x):
	return np.array(x > 0, dtype=np.int)


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
	return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
	return np.maximum(0, x)


def relu_grad(x):
	grad = np.zeros(x)
	grad[x >= 0] = 1
	return grad


def softmax(x):
	if x.ndim == 2:
		x = x.T
		x = x - np.max(x, axis=0)
		y = np.exp(x) / np.sum(np.exp(x), axis=0)
		return y.T

	x = x - np.max(x)  # 오버플로 대책
	return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
	return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)

	# 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
	if t.size == y.size:
		t = t.argmax(axis=1)

	batch_size = y.shape[0]
	return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
	y = softmax(X)
	return cross_entropy_error(y, t)


# test_num 개의 임의의 데이터 셋을 생성한다.
# input_size 만큼의 난수를 생성
def random_data(input_size, label_size, test_num):
	test_data = []
	# test_data = np.random.randint(0, 256, size=(input_size))
	target_data = []
	for i in range(0, test_num):
		data = np.random.randint(0, 256, size=(input_size))
		data = data / 255
		test_data.append(data)
		# label 을 label_size 수만큼 만든다.
		target = [0 for i in range(label_size)]
		target[int(i % (label_size - 1))] = 1
		target_data.append(target)
		# test_data = np.concatenate((test_data, data), axis=0)
	return np.array(test_data,dtype='float64'), np.array(target_data, dtype='float64')
