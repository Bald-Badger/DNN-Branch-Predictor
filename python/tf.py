import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pickle


pc_len = 32
target_len = 32
ghr_len = 256
ga_len = 8
ga_num = 36
entry_cnt = 0  # to be defined by compileData()
entry_len = pc_len + target_len + ghr_len + (ga_len * ga_num)
scale = 64  # for more accurate FP computation.
batch_size = 72
dev = ""


'''
int to binary text, append with tail
e.g. 3 returns 11
     2 returns 10
then each entry in list is scaled by a factor S (for more accurate floating point)
'''
def get_bin(number, l, S=64):
	rep = list(np.binary_repr(number))
	arr = np.zeros(l, dtype=np.uint32)
	for i in range(l):
		if i >= len(rep):
			break
		index = -(i + 1)
		arr[index] = np.int32(rep[index]) * S  # rep index err
	return arr


def compileData(file):
	f = open(file, 'r')
	history = []
	for i in f:
		entryStr = str(i)
		splitEntry = entryStr.split()
		pc = int(splitEntry[3][4:-1])
		target = int(splitEntry[4][8:-1])
		taken = int(splitEntry[5][-2:-1])
		history.append([pc, target, taken])
	f.close()
	entry_cnt = len(history)
	history = np.asarray(history)
	history = np.asarray(history, dtype=np.uint32)

	dataSet = np.zeros(shape=(entry_cnt - ghr_len, entry_len), dtype=np.float32)
	lable = np.zeros(shape=entry_cnt - ghr_len, dtype=np.float32)
	for i in range(ghr_len + 1, entry_cnt):  # strat from ghr_len to collect enough history
		dataEntry = []
		pc = get_bin(number=history[i][0], l=pc_len, S=scale)
		target = get_bin(number=history[i][1], l=target_len, S=scale)
		ghr = history[:, 2][i - ghr_len - 1: i - 1]
		ghr = np.copy(ghr)
		for j in range(len(ghr)):
			ghr[j] = ghr[j] * scale
		ga = []
		for j in range(ga_num):
			ga_entry = get_bin(number=history[i - j][0], l=ga_len, S=scale)
			ga.extend(ga_entry)
		dataEntry.extend(pc)
		dataEntry.extend(target)
		dataEntry.extend(ghr)
		dataEntry.extend(ga)
		dataEntry = np.asarray(dataEntry, dtype=np.float32)
		dataSet[i - ghr_len - 1] = dataEntry
		lable[i - ghr_len - 1] = history[i][2] * scale
	return dataSet, lable


def build_dense_model():
	model = tf.keras.Sequential()
	d1 = layers.Dense(1024, activation='sigmoid')
	d2 = layers.Dense(2048, activation='sigmoid')
	d3 = layers.Dense(6666, activation='sigmoid')
	d4 = layers.Dense(2333, activation='sigmoid')
	d5 = layers.Dense(512, activation='sigmoid')
	d6 = layers.Dense(36, activation='sigmoid')
	d7 = layers.Dense(1, activation='sigmoid')
	model.add(d1)
	model.add(d2)
	model.add(d3)
	model.add(d4)
	model.add(d5)
	model.add(d6)
	model.add(d7)
	return model


if __name__ == '__main__':
	# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


	dataSet, label = compileData('trace2.txt')
	pk_data = (dataSet, label)
	f = open('data.pkl', 'wb')
	pickle.dump(pk_data, f)
	f.close()


	f = open('data.pkl', 'rb')
	dataSet, label = pickle.load(f)
	f.close()

	dataSet = tf.convert_to_tensor(dataSet)
	dataSet = tf.expand_dims(dataSet, 1)
	label = tf.convert_to_tensor(label)
	print(dataSet.shape)
	print(label.shape)
	train_dataset = tf.data.Dataset.from_tensor_slices((dataSet, label))
	train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
	model = build_dense_model()
	model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1), loss='mse')
	model.fit(train_dataset, epochs=30, batch_size=batch_size)

	dataTest, labelTest = compileData('trace.txt')
	dataTest = tf.convert_to_tensor(dataTest)
	dataTest = tf.expand_dims(dataTest, 1)
	labelTest = tf.convert_to_tensor(labelTest)
	# label = tf.expand_dims(label, 1)
	print(dataTest.shape)
	print(labelTest.shape)
	test_dataset = tf.data.Dataset.from_tensor_slices((dataTest, labelTest))
	test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

	f = open('model.mod', 'wb')
	pickle.dump(model, f)
	f.close()

	print("Evaluate")
	result = model.evaluate(test_dataset)
	dict(zip(model.metrics_names, result))

