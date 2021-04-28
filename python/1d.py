import numpy as np
import torch as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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

	if dev == "cuda:0":
		return T.tensor(dataSet).cuda(), T.tensor(lable).cuda()
	else:
		return T.tensor(dataSet), T.tensor(lable)


class BpDataset (Dataset):
	def __init__(self, file):
		(self.data, self.label) = compileData(file)
		print(f"DataSet constructed, {self.data.__len__()} samples")
		print(f"datased shape: {str(self.data.shape)}")

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, idx):
		return self.data[idx], self.label[idx]


class CNNModel(nn.Module):
	def __init__(self):
		super(CNNModel, self).__init__()
		self.d1 = nn.Linear(entry_len, 1024)
		self.d2 = nn.Linear(1024, 2048)
		self.d3 = nn.Linear(2048, 6666)
		self.d4 = nn.Linear(6666, 2333)
		self.d5 = nn.Linear(2333, 1024)
		self.d6 = nn.Linear(1024, 256)
		self.d7 = nn.Linear(256, 1)

	def forward(self, input):
		x = self.d1(input)
		x = T.sigmoid(x)
		x = self.d2(x)
		x = T.sigmoid(x)
		x = self.d3(x)
		x = T.sigmoid(x)
		x = self.d4(x)
		x = T.sigmoid(x)
		x = self.d5(x)
		x = T.sigmoid(x)
		x = self.d6(x)
		x = T.sigmoid(x)
		x = self.d7(x)
		return x


def train(epoch: int, train_dl: DataLoader):
	# get around a problem
	# T.backends.cudnn.enabled = False

	if dev == "cuda:0":
		model = CNNModel().cuda(0)
		print("building model on GPU...")
	else:
		model = CNNModel()
		print("building model on CPU...")
	criterion = nn.MSELoss()
	optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
	running_loss = 0.0
	for epoch in range(epoch):
		for i, (inputs, targets) in enumerate(train_dl):
			#print(inputs.shape)
			#print(targets.shape)
			optimizer.zero_grad()
			yhat = model(inputs)
			loss = criterion(yhat, targets.unsqueeze(1))
			loss.backward()
			optimizer.step()

		# print statistics
		running_loss += loss.item()
		print(f"iter #{i}, current loss: {running_loss}")
	return model


if __name__ == '__main__':
	dataset, label = compileData('trace.txt')
	if T.cuda.is_available():
		dev = "cuda:0"
		print("GPU available")
		device = T.device("cuda:0")
	else:
		dev = "cpu"
		device = T.device("cpu")
	# dev = "cpu"

	data_path = "data_torch.pkl"

	'''
	# construct and save dataset
	train_ds = BpDataset("test.txt")
	data_file = open(data_path, 'wb')
	T.save(train_ds, data_file)
	data_file.close() 
	'''

	#  load dataset
	print("constructing data set...")
	data_file = open(data_path, 'rb')
	train_ds = T.load(data_file, map_location=dev)
	train_dl = DataLoader(train_ds, batch_size=batch_size)
	print('finish construct data set')
	path = "model.mod"

	# train and save model
	print("training model")
	modle = train(epoch=39, train_dl=train_dl)
	T.save(modle.state_dict(), path)

	# load model
	modle = CNNModel()
	device = T.device('cpu')
	modle.load_state_dict(T.load(path, map_location='cpu'))
	dataiter = iter(train_dl)
	data, label = dataiter.next()
	data = data.to(device)
	print(len(label))
	out = modle(data)
	print(out.data)
	print(label)

