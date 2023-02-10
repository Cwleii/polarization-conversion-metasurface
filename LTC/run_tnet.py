import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
import numpy as np
import time
import os

dtype = torch.float
device = torch.device("cpu")

#DataSize = 10
#TrainingDataSize = int(DataSize)
#size = 10

TrainingDataRatio = 0.9
DataSize = 256
TrainingDataSize = int(DataSize * TrainingDataRatio)
TestingDataSize = DataSize - TrainingDataSize
size = int(TestingDataSize)

StartWL = 6.75
EndWL = 16.76
Resolution = 0.05
WL = np.arange(StartWL, EndWL, Resolution)
data = h5py.File('data/data.mat', 'r')
params_min = torch.tensor([1.5, 0.5, 5.0, 1.25])#w, g, r2, t2
params_max = torch.tensor([3.0, 2.0, 7.4, 2.0])
InputNum = WL.size
Trans_data = torch.tensor(data['pcr'][:, 0:DataSize], device=device, dtype=dtype).t()
Params_data = torch.tensor(data['parameter'][:, 0:DataSize], device=device, dtype=dtype).t()
idx = torch.randperm(TestingDataSize)
#idx = np.random.choice(TrainingDataSize, size)
#idx = [1, 79, 194, 14, 71, 175, 200, 174, 58, 13, 87, 99, 219, 220, 35, 24, 151, 4, 111, 182, 0, 5, 8, 201, 78]
#idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Trans_data = Trans_data[idx, :]
Params_data = Params_data[idx, :]
Params_train = Params_data[0:TestingDataSize, :]
Trans_train = Trans_data[0:TestingDataSize, :]
#Params_test = Params_data[TrainingDataSize:TrainingDataSize+TestingDataSize, :]
#Trans_test = Trans_data[TrainingDataSize:TrainingDataSize+TestingDataSize, :]
assert InputNum == Trans_train.shape[1]

'''del data, Input_data, Output_data
print(Trans_data.shape)
print(Params_test.shape)
print(Trans_test.shape)
print(Params_test.shape)
print(Trans_data[0, :])'''

folder_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
path = 'nets/run-tnet/' + folder_name + '/'
os.makedirs(path, exist_ok=True)
log_file = open(path + 'TrainingLog.txt', 'w+')

net = torch.load('nets/rnet/20230210_210341/rnet.pkl')
net = net.to(device)
net.eval()
fnet_path = 'nets/fnet/20220709_185914/fnet.pkl'
fnet = torch.load(fnet_path)
OutputNum = fnet.state_dict()['0.weight'].data.size(1)
fnet.to(device)
fnet.eval()

LossFcn = nn.MSELoss(reduction='mean')

para = []
time_start = time.time()
loss = []
for i in range(0, size):

    InputBatch = Trans_train[i, :].to(device)
    InputParam = Params_train[i, :].to(device)
    Out_Param = (params_max - params_min).to(device).float() * net(InputBatch.unsqueeze(0)).squeeze(0) + params_min.to(device).float()
    Out_Param_no_grad = Out_Param.detach()
    Out_temp = fnet(Out_Param.unsqueeze(0)).squeeze(0)
    FigureTrainLoss = LossFcn(InputBatch, Out_temp)
    time_train = time.time() - time_start
    loss.append(FigureTrainLoss)

loss_test = torch.stack(loss)
loss_test = loss_test.detach().numpy()
loss_avg = np.mean(loss_test)

print('loss_test:', loss_test)
print('loss_test:', loss_test, file=log_file)
print('loss_avg:', loss_avg)
print('loss_avg:', loss_avg, file=log_file)
plt.figure()
plt.axhline(loss_avg, c='red')
plt.scatter(np.arange(0, size, 1), loss_test)
plt.legend(['average_MSE', 'testing_MSE'], loc='best')
plt.savefig(path + 'test_loss')
plt.show()

time_start = time.time()
Out_Param = (params_max - params_min).to(device).float() * net(Trans_train[0, :].unsqueeze(0)).squeeze(0) + params_min.to(device).float()
Out_Param_no_grad = Out_Param.detach()
Out_temp = fnet(Out_Param.unsqueeze(0)).squeeze(0)
time_train = time.time() - time_start
Loss1 = LossFcn(Out_temp, Trans_train[0, :])
print('Structure parameters of curve in figure \'test1.png\':')
print('Structure parameters of curve in figure \'test1.png\':', file=log_file)
print(Params_train[0, :])
print(Params_train[0, :], file=log_file)
print('Designed parameters of curve in figure \'test1.png\':')
print('Designed parameters of curve in figure \'test1.png\':', file=log_file)
print(Out_Param)
print(Out_Param, file=log_file)
plt.plot(WL.T, Trans_train[0, :].cpu().numpy())
plt.plot(WL.T, Out_temp.detach().cpu().numpy())
plt.legend(['GT_PCR', 'Pred_PCR'], loc='lower right')
plt.ylim(0, 1)
plt.savefig(path + 'test1')
plt.show()

Out_Param = (params_max - params_min).to(device).float() * net(Trans_train[8, :].unsqueeze(0)).squeeze(0) + params_min.to(device).float()
Out_Param_no_grad = Out_Param.detach()
Out_temp = fnet(Out_Param.unsqueeze(0)).squeeze(0)
Loss2 = LossFcn(Out_temp, Trans_train[8, :])
print('Structure parameters of curve in figure \'test2.png\':')
print('Structure parameters of curve in figure \'test2.png\':', file=log_file)
print(Params_train[8, :])
print(Params_train[8, :], file=log_file)
print('Designed parameters of curve in figure \'test2.png\':')
print('Designed parameters of curve in figure \'test2.png\':', file=log_file)
print(Out_Param)
print(Out_Param, file=log_file)
plt.plot(WL.T, Trans_train[8, :].cpu().numpy())
plt.plot(WL.T, Out_temp.detach().cpu().numpy())
plt.legend(['GT_PCR', 'Pred_PCR'], loc='lower right')
plt.ylim(0, 1)
plt.savefig(path + 'test2')
plt.show()


Out_Param = (params_max - params_min).to(device).float() * net(Trans_train[16, :].unsqueeze(0)).squeeze(0) + params_min.to(device).float()
Out_Param_no_grad = Out_Param.detach()
Out_temp = fnet(Out_Param.unsqueeze(0)).squeeze(0)
Loss3 = LossFcn(Out_temp, Trans_train[16, :])
print('Structure parameters of curve in figure \'test3.png\':')
print('Structure parameters of curve in figure \'test3.png\':', file=log_file)
print(Params_train[16, :])
print(Params_train[16, :], file=log_file)
print('Designed parameters of curve in figure \'test3.png\':')
print('Designed parameters of curve in figure \'test3.png\':', file=log_file)
print(Out_Param)
print(Out_Param, file=log_file)
plt.plot(WL.T, Trans_train[16, :].cpu().numpy())
plt.plot(WL.T, Out_temp.detach().cpu().numpy())
plt.legend(['GT_PCR', 'Pred_PCR'], loc='lower right')
plt.ylim(0, 1)
plt.savefig(path + 'test3')
plt.show()


Out_Param = (params_max - params_min).to(device).float() * net(Trans_train[24, :].unsqueeze(0)).squeeze(0) + params_min.to(device).float()
Out_Param_no_grad = Out_Param.detach()
Out_temp = fnet(Out_Param.unsqueeze(0)).squeeze(0)
Loss4 = LossFcn(Out_temp, Trans_train[24, :])
print('Structure parameters of curve in figure \'test4.png\':')
print('Structure parameters of curve in figure \'test4.png\':', file=log_file)
print(Params_train[24, :])
print(Params_train[24, :], file=log_file)
print('Designed parameters of curve in figure \'test4.png\':')
print('Designed parameters of curve in figure \'test4.png\':', file=log_file)
print(Out_Param)
print(Out_Param, file=log_file)
plt.plot(WL.T, Trans_train[24, :].cpu().numpy())
plt.plot(WL.T, Out_temp.detach().cpu().numpy())
plt.legend(['GT_PCR', 'Pred_PCR'], loc='lower right')
plt.ylim(0, 1)
plt.savefig(path + 'test4')
plt.show()

print('Running finished!', '| loss1: %.5f' % Loss1.data.item(), '| loss2: %.5f' % Loss2.data.item(),
      '| loss3: %.5f' % Loss3.data.item(), '| loss4: %.5f' % Loss4.data.item())
print('Running finished!', '| loss1: %.5f' % Loss1.data.item(), '| loss2: %.5f' % Loss2.data.item(),
      '| loss3: %.5f' % Loss3.data.item(), '| loss4: %.5f' % Loss4.data.item(), file=log_file)
print('Running time on training set: %.8fs' % time_train)
print('Running time on training set: %.8fs' % time_train, file=log_file)
log_file.close()

'''
for i in range(0, TestingDataSize):
    InputBatch = Trans_train[i, :].to(device)
    InputParam = Params_train[i, :].to(device)
    Out_Param = (params_max - params_min).to(device).float() * net(InputBatch.unsqueeze(0)).squeeze(0) + params_min.to(device).float()
    Out_temp = fnet(Out_Param.unsqueeze(0)).squeeze(0)
    FigureTrainLoss = LossFcn(InputBatch, Out_temp)
    print('Structure parameters :')
    print('Structure parameters :', file=log_file)
    print(InputParam)
    print(InputParam, file=log_file)
    print('Designed parameters :')
    print('Designed parameters :', file=log_file)
    print(Out_Param)
    print(Out_Param, file=log_file)
    print('loss : %.5f' % FigureTrainLoss.data.item())
    print('loss : %.5f' % FigureTrainLoss.data.item(), file=log_file)


b = 4
time_start = time.time()
Params_temp = (params_max - params_min).to(device).float() * net(Trans_train[b, :].to(device).unsqueeze(0)).squeeze(0) + params_min.to(device).float()
Output_temp = fnet(Params_temp.unsqueeze(0)).squeeze(0)
time_train = time.time() - time_start
FigureTrainLoss = LossFcn(Trans_train[b, :].to(device), Output_temp)
print('Structure parameters of curve in figure \'test1.png\':')
print('Structure parameters of curve in figure \'test1.png\':', file=log_file)
print(Params_train[b, :])
print(Params_train[b, :], file=log_file)
print('Designed parameters of curve in figure \'test1.png\':')
print('Designed parameters of curve in figure \'test1.png\':', file=log_file)
print(Params_temp)
print(Params_temp, file=log_file)
#print(Output_temp)
print(Output_temp, file=log_file)
plt.figure()
plt.plot(WL.T, Trans_train[b, :].cpu().numpy())
plt.plot(WL.T, Output_temp.detach().cpu().numpy())
plt.xlabel("Wavelength(nm)")
plt.ylabel("PCR")
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='best')
plt.savefig(path + 'test1')
plt.show()
#print('Running time on training set: %.8fs' % time_train)
#print('Running time on training set: %.8fs' % time_train, file=log_file)
print('| loss in figure : %.5f' % FigureTrainLoss.data.item())
print('| loss in figure : %.5f' % FigureTrainLoss.data.item(), file=log_file)
log_file.close()


for b in range(0, TestingDataSize):
    time_start = time.time()
    Params_temp = (params_max - params_min).to(device).float() * net(Trans_train[b, :].to(device).unsqueeze(0)).squeeze(0) + params_min.to(device).float()
    Output_temp = fnet(Params_temp.unsqueeze(0)).squeeze(0)
    time_train = time.time() - time_start
    FigureTrainLoss = LossFcn(Trans_train[b, :].to(device), Output_temp)

    plt.figure(figsize=(4, 3))
    lines = plt.plot(WL.T, Trans_train[b, :].cpu().numpy(), WL.T, Output_temp.detach().cpu().numpy())
    # plt.plot(WL.T, Output_temp.detach().cpu().numpy())
    # plt.xlabel("Wavelength(nm)")
    # plt.ylabel("PCR")
    plt.tick_params(labelsize=13)
    plt.ylim(0, 1)
    plt.setp(lines, linewidth=3.0)
    # plt.legend(['GT', 'pred'], loc='best')
    plt.savefig(path + 'test-{}'.format(b + 1), dpi=600)
    plt.show()
    print('Running time on training set: %.8fs' % time_train)
    print('Average running time of training sample: %.8fs' % (time_train / TestingDataSize))

    plt.figure()
    plt.plot(WL.T, Trans_train[b, :].cpu().numpy())
    plt.plot(WL.T, Output_temp.detach().cpu().numpy())
    plt.xlabel("Frequency(GHz)")
    plt.ylabel("PCR")
    plt.ylim(0, 1)
    plt.legend(['GT', 'pred'], loc='best')
    plt.savefig(path + 'test-{}'.format(b + 1))
    plt.show()
'''

