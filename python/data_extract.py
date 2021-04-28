import numpy as np
import torch as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle


history = []
L = 32  # 32 entry history
W = 16  # 15 + taken
H = 128  # forward or backward 64

N = 0	# to be defined by compileData()
dev = ""

'''
int to binary text, append with tail
e.g. 3 with tail 0 returns 110
     2 with tail 1 returns 101
'''
def get_bin(number, l, tail):
	rep = list(np.binary_repr(number))
	arr = np.zeros(l + 1, dtype=np.float32)
	for i in range(l):
		if i >= len(rep):
			break
		index = -(i + 2)
		arr[index] = np.float32(rep[index])  # rep index err

	arr[-1] = np.float32(tail)
	return arr


def get_slice(i):
	dataSlice = np.zeros(shape=(H, W), dtype=np.float32)
	offset = round(H / 2) + (history[i][1] - history[i][0])
	if offset > (H - 1):
		offset = (H - 1)
	if offset < 0:
		offset = 0
	# encode the taken bit at last
	dataSlice[offset] = get_bin(history[i][0], (W - 1), history[i][2])
	return dataSlice


def compileData(file):
	f = open(file, 'r')
	for i in f:
		entryStr = str(i)
		splitEntry = entryStr.split()
		pc = int(splitEntry[3][4:-1])
		target = int(splitEntry[4][8:-1])
		taken = int(splitEntry[5][-2:-1])
		history.append([pc, target, taken])
	f.close()

	N = len(history) - L
	# N = 200  # use a reasonable value at first

	dataSet = np.zeros(shape=(N, L, H, W), dtype=np.float32)
	lable = np.zeros(shape=N, dtype=np.float32)
	dataEntry = np.zeros(shape=(L, H, W), dtype=np.float32)

	# first slice
	for i in range(L):
		dataSlice = get_slice(i)
		dataEntry[i] = dataSlice

	# following slice
	for i in range(L + 1, N + L - 1):
		dataSet[i - L] = dataEntry
		dataEntry = dataEntry[1:, :, :]
		try:
			dataSlice = get_slice(i)
		except IndexError:
			pass

		# set the nearest taken bit as lable
		lable[i - L] = history[i][2]
		dataEntry = np.concatenate((dataEntry, [dataSlice]), axis=0)
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
		self.conv1 = nn.Conv2d(L, 256, kernel_size=3, stride=1)
		self.pool1 = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(256, 128, kernel_size=2, stride=1)
		self.pool2 = nn.MaxPool2d(2, 2)
		self.conv3 = nn.Conv2d(128, 64, kernel_size=2, stride=1)
		self.conv4 = nn.Conv2d(64, 32, kernel_size=2, stride=1)
		self.flat = nn.Flatten()
		self.fc1 = nn.Linear(928, 2140)
		self.fc2 = nn.Linear(2140, 720)
		self.fc3 = nn.Linear(720, 1)


	def forward (self, input):
		x = self.conv1(input)
		x = self.pool1(F.relu(x))
		x = self.pool2(F.relu(self.conv2(x)))
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.flat(x)
		x = T.sigmoid(self.fc1(x))
		x = T.sigmoid(self.fc2(x))
		x = T.sigmoid(self.fc3(x))
		return x


def train(epoch: int, train_dl: DataLoader):
	# get around a problem
	# T.backends.cudnn.enabled = False

	if dev == "cuda:0":
		model = CNNModel().cuda(0)
		print("building model on GPU...")
	else:
		model = CNNModel()
	criterion = nn.MSELoss()
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	running_loss = 0.0
	for epoch in range(epoch):
		for i, (inputs, targets) in enumerate(train_dl):
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
	if T.cuda.is_available():
		dev = "cuda:0"
		print("GPU available")
		device = T.device("cuda:0")
	else:
		dev = "cpu"
		device = T.device("cpu")
	# dev = "cpu"


	data_path = "data.pkl"

	# construct and save dataset

	train_ds = BpDataset("trace2.txt")
	data_file = open(data_path, 'wb')
	T.save(train_ds, data_file)
	data_file.close()


	# load dataset
	print("constructing data set...")
	data_file = open(data_path, 'rb')
	train_ds = T.load(data_file, map_location=dev)
	train_dl = DataLoader(train_ds, batch_size=32)
	print('finish construct data set')
	path = "model.mod"
	# train and save model
	print("training model")
	modle = train(epoch=25, train_dl=train_dl)
	T.save(modle.state_dict(), path)

	# load model
	modle = CNNModel()
	device = T.device('cpu')
	modle.load_state_dict(T.load(path, map_location='cpu'))
	dataiter = iter(train_dl)
	data, label = dataiter.next()
	data = data.to(device)
	label = label.to(device)
	out = modle(data)
	print(out.data)
	print(label)
