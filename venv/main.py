import torch
import torch.nn as nn
import numpy as np
import json
import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Dataset(Dataset):

    def __init__(self, *args):  # получаем n массивов данных
        cnt, labels, temp = 0, [], []
        self.data = []
        for data in args:  # выделяем по одному из них
            self.data.extend(
                torch.tensor(data).type(torch.float32))  # преобразуем каждый в тензор и добавляем к общим данным
            for j in range(len(data)):
                labels.append(cnt)  # присовим номера каждому подмассиву для проверки
            cnt += 1
        self.labels = labels

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

f = open('dataset/stop2.txt').readline()
f = json.loads(f)

f = open('dataset/kulaksgat2.txt').readline()
f = json.loads(f)

data, temp = [],[]
gests = os.listdir('dataset')
for gest in gests:
    data = []
    print(gest)
    f = open('dataset/' + str(gest)).readline()
    f = json.loads(f)
    for i in range (len(f)):
        data.append(f[i])
    temp.append(data)
data = np.array(temp, dtype=object)

dataset = Dataset(*data)
dataloader = DataLoader(dataset, shuffle=True, batch_size = 16)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(42, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 32)
        self.fc4 = nn.Linear(32, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for epoch in range(10):
    running_loss = 0.0
    for samples, labels in dataloader:
        pred = net(samples)
        loss = criterion(pred, labels)

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        running_loss += loss.item()
    print(running_loss)
print('Finished Training')

kk = torch.tensor([0.41655007004737854, 0.7089771032333374, 0.4633732736110687, 0.6954310536384583, 0.5125848650932312, 0.6476010084152222, 0.5015125274658203, 0.592243492603302, 0.4613598585128784, 0.5826143026351929, 0.5011842250823975, 0.5836077928543091, 0.509346067905426, 0.5467285513877869, 0.49322056770324707, 0.5981612205505371, 0.48973703384399414, 0.6084542870521545, 0.47082626819610596, 0.5737633109092712, 0.47700440883636475, 0.5433168411254883, 0.46477067470550537, 0.6013384461402893, 0.46333837509155273, 0.599715530872345, 0.4406990706920624, 0.571521520614624, 0.44218024611473083, 0.5451012849807739, 0.436707079410553, 0.6007078289985657, 0.4390093982219696, 0.6013930439949036, 0.4080536961555481, 0.5698967576026917, 0.4114087224006653, 0.549913763999939, 0.41262519359588623, 0.5887411832809448, 0.41415098309516907, 0.5954364538192749])

print(net(kk))

from datetime import datetime

start_time = datetime.now()
for i in range(60):
    print(net(kk))
time.sleep(0)

print(datetime.now() - start_time)
