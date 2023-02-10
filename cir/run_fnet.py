import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
import numpy as np
import time
import os
from matplotlib.pyplot import MultipleLocator

dtype = torch.float
device = torch.device("cpu")

#TrainingDataRatio = 0.9
DataSize = 625
#TrainingDataSize = int(DataSize * TrainingDataRatio)
#TestingDataSize = DataSize - TrainingDataSize
TrainingDataSize = int(DataSize)
size = int(DataSize)

StartWL = 300
EndWL = 1501
Resolution = 3
WL = np.arange(StartWL, EndWL, Resolution)
data = h5py.File('data/pcr625.mat', 'r')
OutputNum = WL.size
Input_data = torch.tensor(data['parameter'][:, 0:DataSize], device=device, dtype=dtype).t()
Output_data = torch.tensor(data['pcr'][:, 0:DataSize], device=device, dtype=dtype).t()
idx = torch.randperm(DataSize)
Input_data = Input_data[idx, :]
Output_data = Output_data[idx, :]
Input_train = Input_data[0:size, :]
Output_train = Output_data[0:size, :]
#Input_test = Input_data[TrainingDataSize:TrainingDataSize + TestingDataSize, :]
#Output_test = Output_data[TrainingDataSize:TrainingDataSize + TestingDataSize, :]
#InputNum = Input_train.shape[1]
assert WL.size == Output_train.shape[1]

'''del data, Input_data, Output_data
print(Input_train.shape)
print(Output_train.shape)
print(Input_test.shape)
print(Output_test.shape)
print(Input_train[0, :])'''

folder_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
path = 'nets/run-fnet/' + folder_name + '/'
os.makedirs(path, exist_ok=True)
log_file = open(path + 'TrainingLog.txt', 'w+')

net = torch.load('nets/fnet/20230129_162229/fnet.pkl')
net = net.to(device)
net.eval()

LossFcn = nn.MSELoss(reduction='mean')

time_start = time.time()
loss = []
for i in range(0, size):

    Input_temp = Output_train[i, :].to(device)
    Input_param = Input_train[i, :].to(device)
    Output_temp = net(Input_param.unsqueeze(0)).squeeze(0)
    time_train = time.time() - time_start
    FinalTrainLoss = LossFcn(Input_temp, Output_temp)
    time_train = time.time() - time_start
    loss.append(FinalTrainLoss)

loss_test = torch.stack(loss)
loss_test = loss_test.detach().numpy()
loss_avg = np.mean(loss_test)

print('loss_test:', loss_test)
print('loss_test:', loss_test, file=log_file)
print('loss_avg:', loss_avg)
print('loss_avg:', loss_avg, file=log_file)
plt.figure()
plt.axhline(loss_avg, c='red')
plt.scatter(np.arange(0, DataSize, 1), loss_test)
plt.legend(['average_MSE', 'testing_MSE'], loc='best')
y_major_locator = MultipleLocator(0.0005)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(-0.0001, 0.0015)
plt.savefig(path + 'test_loss')
plt.show()




time_start = time.time()
Output_temp = net(Input_train[150, :].unsqueeze(0)).squeeze(0)
time_train = time.time() - time_start
Loss1 = LossFcn(Output_train[150, :], Output_temp)
print('Structure parameters of curve in figure \'test1.png\':')
print('Structure parameters of curve in figure \'test1.png\':', file=log_file)
print(Input_train[150, :])
print(Input_train[150, :], file=log_file)
plt.plot(WL.T, Output_train[150, :].cpu().numpy())
plt.plot(WL.T, Output_temp.detach().cpu().numpy())
plt.legend(['GT_PCR', 'Pred_PCR'], loc='lower right')
plt.savefig(path + 'test1')
plt.show()

Output_temp = net(Input_train[300, :].unsqueeze(0)).squeeze(0)
Loss2 = LossFcn(Output_train[300, :], Output_temp)
print('Structure parameters of curve in figure \'test2.png\':')
print('Structure parameters of curve in figure \'test2.png\':', file=log_file)
print(Input_train[300, :])
print(Input_train[300, :], file=log_file)
plt.plot(WL.T, Output_train[300, :].cpu().numpy())
plt.plot(WL.T, Output_temp.detach().cpu().numpy())
plt.legend(['GT_PCR', 'Pred_PCR'], loc='lower right')
plt.savefig(path + 'test2')
plt.show()

Output_temp = net(Input_train[450, :].unsqueeze(0)).squeeze(0)
Loss3 = LossFcn(Output_train[450, :], Output_temp)
print('Structure parameters of curve in figure \'test3.png\':')
print('Structure parameters of curve in figure \'test3.png\':', file=log_file)
print(Input_train[450, :])
print(Input_train[450, :], file=log_file)
plt.plot(WL.T, Output_train[450, :].cpu().numpy())
plt.plot(WL.T, Output_temp.detach().cpu().numpy())
plt.legend(['GT_PCR', 'Pred_PCR'], loc='lower right')
plt.savefig(path + 'test3')
plt.show()

Output_temp = net(Input_train[600, :].unsqueeze(0)).squeeze(0)
Loss4 = LossFcn(Output_train[600, :], Output_temp)
print('Structure parameters of curve in figure \'test4.png\':')
print('Structure parameters of curve in figure \'test4.png\':', file=log_file)
print(Input_train[600, :])
print(Input_train[600, :], file=log_file)
plt.plot(WL.T, Output_train[600, :].cpu().numpy())
plt.plot(WL.T, Output_temp.detach().cpu().numpy())
plt.legend(['GT_PCR', 'Pred_PCR'], loc='lower right')
plt.savefig(path + 'test4')
plt.show()

print('Running finished!', '| loss1: %.5f' % Loss1.data.item(), '| loss2: %.5f' % Loss2.data.item(),
      '| loss3: %.5f' % Loss3.data.item(), '| loss4: %.5f' % Loss4.data.item())
print('Running finished!', '| loss1: %.5f' % Loss1.data.item(), '| loss2: %.5f' % Loss2.data.item(),
      '| loss3: %.5f' % Loss3.data.item(), '| loss4: %.5f' % Loss4.data.item(), file=log_file)
print('Running time on training set: %.8fs' % time_train)
print('Running time on training set: %.8fs' % time_train, file=log_file)
log_file.close()


