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

DataSize = 20
TrainingDataSize = int(DataSize)
size = 20

StartWL = 500
EndWL = 801
Resolution = 2
WL = np.arange(StartWL, EndWL, Resolution)
data = h5py.File('data/test.mat', 'r')
params_min = torch.tensor([150, 50, 40, 50])#l, t, t2, w
params_max = torch.tensor([300, 110, 100, 140])
InputNum = WL.size
Trans_data = torch.tensor(data['pcr'][:, 0:DataSize], device=device, dtype=dtype).t()
Params_data = torch.tensor(data['para'][:, 0:DataSize], device=device, dtype=dtype).t()
#idx = torch.randperm(DataSize)
#idx = np.random.choice(TrainingDataSize, size)
#idx = [1, 79, 194, 14, 71, 175, 200, 174, 58, 13, 87, 99, 219, 220, 35, 24, 151, 4, 111, 182]
idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
Trans_data = Trans_data[idx, :]
Params_data = Params_data[idx, :]
Params_train = Params_data[0:size, :]
Trans_train = Trans_data[0:size, :]
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

net = torch.load('nets/rnet/adabelief-batch64/rnet.pkl')
net = net.to(device)
net.eval()
fnet_path = 'nets/fnet/batch64-adam/fnet.pkl'
fnet = torch.load(fnet_path)
OutputNum = fnet.state_dict()['0.weight'].data.size(1)
fnet.to(device)
fnet.eval()

LossFcn = nn.MSELoss(reduction='mean')


for i in range(0, size):
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

'''
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

'''
for b in range(0, 20):
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

    '''plt.figure()
    plt.plot(WL.T, Trans_train[b, :].cpu().numpy())
    plt.plot(WL.T, Output_temp.detach().cpu().numpy())
    plt.xlabel("Wavelength(nm)")
    plt.ylabel("PCR")
    plt.ylim(0, 1)
    plt.legend(['GT', 'pred'], loc='best')
    plt.savefig(path + 'test-{}'.format(b + 1))
    plt.show()'''

    log_file.close()
