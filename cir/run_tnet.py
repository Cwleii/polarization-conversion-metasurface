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

DataSize = 625
TrainingDataSize = int(DataSize)
size = int(DataSize)

StartWL = 300
EndWL = 1501
Resolution = 3
WL = np.arange(StartWL, EndWL, Resolution)
data = h5py.File('data/pcr625.mat', 'r')
params_min = torch.tensor([50, 70, 50, 10, 20])# (L1, L2, t1, t2, W)
params_max = torch.tensor([80, 110, 150, 30, 60])
InputNum = WL.size
Trans_data = torch.tensor(data['pcr'][:, 0:DataSize], device=device, dtype=dtype).t()
Params_data = torch.tensor(data['parameter'][:, 0:DataSize], device=device, dtype=dtype).t()
idx = torch.randperm(DataSize)
#idx = np.random.choice(TrainingDataSize, size)
#idx = [1, 79, 194, 14, 71, 175, 200, 174, 58, 13, 87, 99, 219, 220, 35, 24, 151, 4, 111, 182]
#idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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

net = torch.load('nets/rnet/20230129_200410/rnet.pkl')
net = net.to(device)
net.eval()
fnet_path = 'nets/fnet/20230129_162229/fnet.pkl'
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
plt.scatter(np.arange(0, DataSize, 1), loss_test)
plt.legend(['average_MSE', 'testing_MSE'], loc='best')
plt.savefig(path + 'test_loss')
plt.show()

time_start = time.time()
Out_Param = (params_max - params_min).to(device).float() * net(Trans_train[150, :].unsqueeze(0)).squeeze(0) + params_min.to(device).float()
Out_Param_no_grad = Out_Param.detach()
Out_temp = fnet(Out_Param.unsqueeze(0)).squeeze(0)
time_train = time.time() - time_start
Loss1 = LossFcn(Out_temp, Trans_train[150, :])
print('Structure parameters of curve in figure \'test1.png\':')
print('Structure parameters of curve in figure \'test1.png\':', file=log_file)
print(Params_train[150, :])
print(Params_train[150, :], file=log_file)
print('Designed parameters of curve in figure \'test1.png\':')
print('Designed parameters of curve in figure \'test1.png\':', file=log_file)
print(Out_Param)
print(Out_Param, file=log_file)
plt.plot(WL.T, Trans_train[150, :].cpu().numpy())
plt.plot(WL.T, Out_temp.detach().cpu().numpy())
plt.legend(['GT_PCR', 'Pred_PCR'], loc='lower right')
plt.savefig(path + 'test1')
plt.show()

Out_Param = (params_max - params_min).to(device).float() * net(Trans_train[300, :].unsqueeze(0)).squeeze(0) + params_min.to(device).float()
Out_Param_no_grad = Out_Param.detach()
Out_temp = fnet(Out_Param.unsqueeze(0)).squeeze(0)
Loss2 = LossFcn(Out_temp, Trans_train[300, :])
print('Structure parameters of curve in figure \'test2.png\':')
print('Structure parameters of curve in figure \'test2.png\':', file=log_file)
print(Params_train[300, :])
print(Params_train[300, :], file=log_file)
print('Designed parameters of curve in figure \'test2.png\':')
print('Designed parameters of curve in figure \'test2.png\':', file=log_file)
print(Out_Param)
print(Out_Param, file=log_file)
plt.plot(WL.T, Trans_train[300, :].cpu().numpy())
plt.plot(WL.T, Out_temp.detach().cpu().numpy())
plt.legend(['GT_PCR', 'Pred_PCR'], loc='lower right')
plt.savefig(path + 'test2')
plt.show()


Out_Param = (params_max - params_min).to(device).float() * net(Trans_train[450, :].unsqueeze(0)).squeeze(0) + params_min.to(device).float()
Out_Param_no_grad = Out_Param.detach()
Out_temp = fnet(Out_Param.unsqueeze(0)).squeeze(0)
Loss3 = LossFcn(Out_temp, Trans_train[450, :])
print('Structure parameters of curve in figure \'test3.png\':')
print('Structure parameters of curve in figure \'test3.png\':', file=log_file)
print(Params_train[300, :])
print(Params_train[300, :], file=log_file)
print('Designed parameters of curve in figure \'test3.png\':')
print('Designed parameters of curve in figure \'test3.png\':', file=log_file)
print(Out_Param)
print(Out_Param, file=log_file)
plt.plot(WL.T, Trans_train[450, :].cpu().numpy())
plt.plot(WL.T, Out_temp.detach().cpu().numpy())
plt.legend(['GT_PCR', 'Pred_PCR'], loc='lower right')
plt.savefig(path + 'test3')
plt.show()


Out_Param = (params_max - params_min).to(device).float() * net(Trans_train[600, :].unsqueeze(0)).squeeze(0) + params_min.to(device).float()
Out_Param_no_grad = Out_Param.detach()
Out_temp = fnet(Out_Param.unsqueeze(0)).squeeze(0)
Loss4 = LossFcn(Out_temp, Trans_train[600, :])
print('Structure parameters of curve in figure \'test4.png\':')
print('Structure parameters of curve in figure \'test4.png\':', file=log_file)
print(Params_train[600, :])
print(Params_train[600, :], file=log_file)
print('Designed parameters of curve in figure \'test4.png\':')
print('Designed parameters of curve in figure \'test4.png\':', file=log_file)
print(Out_Param)
print(Out_Param, file=log_file)
plt.plot(WL.T, Trans_train[600, :].cpu().numpy())
plt.plot(WL.T, Out_temp.detach().cpu().numpy())
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

'''print('Structure parameters :')
    print('Structure parameters :', file=log_file)
    print(InputParam)
    print(InputParam, file=log_file)
    print('Designed parameters :')
    print('Designed parameters :', file=log_file)
    print(Out_Param)
    print(Out_Param, file=log_file)
    print('loss : %.5f' % FigureTrainLoss.data.item())
    print('loss : %.5f' % FigureTrainLoss.data.item(), file=log_file)
    para.append(Out_Param_no_grad)

    print('Running time on training set: %.8fs' % time_train)
    print('Running time on training set: %.8fs' % time_train, file=log_file)
    #print(para)'''
'''
Params_GT = Params_train.numpy()
Params_Test = torch.stack(para)
Params_Test = Params_Test.numpy()
print('Params_Test:', Params_Test)
print('Params_Test:', Params_Test, file=log_file)
print('Params_GT:', Params_GT)
print('Params_GT:', Params_GT, file=log_file)
for j in range(0, 5):
    x = Params_GT[:, j]
    y = Params_Test[:, j]
    plt.figure()
    plt.scatter(x, y, s=10)
    plt.xlabel("GT(nm)")
    plt.ylabel("Pred_net(nm)")
    plt.savefig(path + 'test-{}'.format(j + 1), dpi=600)
    plt.show()
'''
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
'''
for b in range(0, 2):
    time_start = time.time()
    Params_temp = (params_max - params_min).to(device).float() * net(Trans_train[b, :].to(device).unsqueeze(0)).squeeze(0) + params_min.to(device).float()
    Output_temp = fnet(Params_temp.unsqueeze(0)).squeeze(0)
    time_train = time.time() - time_start

    FigureTrainLoss = LossFcn(Trans_train[b, :].to(device), Output_temp)


    plt.figure(figsize=(4, 3))
    lines = plt.plot(WL.T, Trans_train[b, :].cpu().numpy(), WL.T, Output_temp.detach().cpu().numpy())
   # plt.plot(WL.T, Output_temp.detach().cpu().numpy())
    #plt.xlabel("Wavelength(nm)")
    #plt.ylabel("PCR")
    plt.tick_params(labelsize=13)
    plt.ylim(0, 1)
    plt.setp(lines, linewidth=3.0 )
    #plt.legend(['GT', 'pred'], loc='best')
    plt.savefig(path + 'test-{}'.format(b + 1), dpi=600)
    plt.show()

    log_file.close()'''
