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

TrainingDataRatio = 0.1
DataSize = 256
#DataSize = 4 ** 4
TrainingDataSize = int(DataSize * TrainingDataRatio)
#TestingDataSize = DataSize - TrainingDataSize

#TrainingDataSize = int(DataSize)
size = int(TrainingDataSize)

StartWL = 6.75
EndWL = 16.76
Resolution = 0.05
WL = np.arange(StartWL, EndWL, Resolution)
data = h5py.File('data/data.mat', 'r')
OutputNum = WL.size
Input_data = torch.tensor(data['parameter'][:, 0:DataSize], device=device, dtype=dtype).t()
Output_data = torch.tensor(data['pcr'][:, 0:DataSize], device=device, dtype=dtype).t()
idx = torch.randperm(DataSize)
Input_data = Input_data[idx, :]
Output_data = Output_data[idx, :]
Input_train = Input_data[0:TrainingDataSize, :]
Output_train = Output_data[0:TrainingDataSize, :]
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

net = torch.load('nets/fnet/20220309_104056/fnet.pkl')
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
plt.scatter(np.arange(0, TrainingDataSize, 1), loss_test)
plt.legend(['average_MSE', 'testing_MSE'], loc='best')
#y_major_locator = MultipleLocator(0.0005)
#ax = plt.gca()
#ax.yaxis.set_major_locator(y_major_locator)
#plt.xlim(0, 26)
plt.ylim(-0.0001, 0.001)
plt.savefig(path + 'test_loss')
plt.show()




time_start = time.time()
Output_temp = net(Input_train[0, :].unsqueeze(0)).squeeze(0)
time_train = time.time() - time_start
Loss1 = LossFcn(Output_train[0, :], Output_temp)
print('Structure parameters of curve in figure \'test1.png\':')
print('Structure parameters of curve in figure \'test1.png\':', file=log_file)
print(Input_train[0, :])
print(Input_train[0, :], file=log_file)
plt.plot(WL.T, Output_train[0, :].cpu().numpy())
plt.plot(WL.T, Output_temp.detach().cpu().numpy())
plt.legend(['GT_PCR', 'Pred_PCR'], loc='lower right')
plt.savefig(path + 'test1')
plt.show()

Output_temp = net(Input_train[8, :].unsqueeze(0)).squeeze(0)
Loss2 = LossFcn(Output_train[8, :], Output_temp)
print('Structure parameters of curve in figure \'test2.png\':')
print('Structure parameters of curve in figure \'test2.png\':', file=log_file)
print(Input_train[8, :])
print(Input_train[8, :], file=log_file)
plt.plot(WL.T, Output_train[8, :].cpu().numpy())
plt.plot(WL.T, Output_temp.detach().cpu().numpy())
plt.legend(['GT_PCR', 'Pred_PCR'], loc='lower right')
plt.savefig(path + 'test2')
plt.show()

Output_temp = net(Input_train[16, :].unsqueeze(0)).squeeze(0)
Loss3 = LossFcn(Output_train[16, :], Output_temp)
print('Structure parameters of curve in figure \'test3.png\':')
print('Structure parameters of curve in figure \'test3.png\':', file=log_file)
print(Input_train[16, :])
print(Input_train[16, :], file=log_file)
plt.plot(WL.T, Output_train[16, :].cpu().numpy())
plt.plot(WL.T, Output_temp.detach().cpu().numpy())
plt.legend(['GT_PCR', 'Pred_PCR'], loc='lower right')
plt.savefig(path + 'test3')
plt.show()

Output_temp = net(Input_train[24, :].unsqueeze(0)).squeeze(0)
Loss4 = LossFcn(Output_train[24, :], Output_temp)
print('Structure parameters of curve in figure \'test4.png\':')
print('Structure parameters of curve in figure \'test4.png\':', file=log_file)
print(Input_train[24, :])
print(Input_train[24, :], file=log_file)
plt.plot(WL.T, Output_train[24, :].cpu().numpy())
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

'''time_start = time.time()
Output_temp = net(Input_train)
time_train = time.time() - time_start
FinalTrainLoss = LossFcn(Output_train, Output_temp)
plt.plot(WL.T, Output_train[0, :].cpu().numpy())
plt.plot(WL.T, Output_temp[0, :].detach().cpu().numpy())
plt.legend(['GT', 'pred'], loc='lower right')
plt.show()
time_start = time.time()
Output_temp = net(Input_test)
time_test = time.time() - time_start
FinalTestLoss = LossFcn(Output_test, Output_temp)
plt.plot(WL.T, Output_test[15, :].cpu().numpy())
plt.plot(WL.T, Output_temp[15, :].detach().cpu().numpy())
plt.legend(['GT', 'pred'], loc='lower right')
plt.show()
print('Running finished!', '| train loss: %.5f' % FinalTrainLoss.data.item(),
      '| test loss: %.5f' % FinalTestLoss.data.item())
print('Running time on training set: %.8fs' % time_train)
print('Average running time of training sample: %.8fs' % (time_train / TrainingDataSize))
print('Running time on testing set: %.8fs' % time_test)
print('Average running time of testing sample: %.8fs' % (time_test / TestingDataSize))'''
