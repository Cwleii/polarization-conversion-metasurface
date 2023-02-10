import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
import numpy as np
import math
import time
import os
import sys
from adabelief_pytorch import AdaBelief

dtype = torch.float
device_data = torch.device('cpu')
device_train = torch.device('cuda:0')
device_test = torch.device('cpu')

Material = 'Meta'
# Material = 'TF'
TrainingDataRatio = 0.9
DataSize = 256
TrainingDataSize = int(DataSize * TrainingDataRatio)
TestingDataSize = DataSize - TrainingDataSize
BatchSize = 32
BatchEnable = True
EpochNum = 1001
TestInterval = 10
lr = 1e-3
lr_decay_step = 50
lr_decay_gamma = 0.8
if Material == 'Meta':
    params_min = torch.tensor([1.6, 0.4, 4.9, 1.24])#w, g, r2, t2([1.5, 0.5, 5.0, 1.25])
    params_max = torch.tensor([3.1, 2.1, 7.5, 1.9])#[3.0, 2.0, 7.4, 2.0]
else:
    params_min = torch.tensor([100])
    params_max = torch.tensor([300])

folder_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
path = 'nets/rnet/' + folder_name + '/'

if Material == 'Meta':
    fnet_path = 'nets/fnet/20220709_185914/fnet.pkl'
else:
    fnet_path = 'nets/fnet/TF_100-300nm/fnet.pkl'
    # fnet_path = 'nets/fnet/TF_0-150nm/fnet.pkl'

if Material == 'Meta':
    data = h5py.File('data/data.mat', 'r')
    StartWL = 6.75
    EndWL = 16.76
    Resolution = 0.05
    WL = np.arange(StartWL, EndWL, Resolution)
    InputNum = WL.size
    Params_data = torch.tensor(data['parameter'][:, 0:DataSize], device=device_data, dtype=dtype).t()
    Trans_data = torch.tensor(data['pcr'][:, 0:DataSize], device=device_data, dtype=dtype).t()
    idx = torch.randperm(DataSize)
    Params_data = Params_data[idx, :]
    Trans_data = Trans_data[idx, :]
    Params_train = Params_data[0:DataSize, :]
    Trans_train = Trans_data[0:DataSize, :]
    #Params_train = Params_data[0:TrainingDataSize, :]
    #Trans_train = Trans_data[0:TrainingDataSize, :]
    Params_test = Params_data[TrainingDataSize:TrainingDataSize+TestingDataSize, :]
    Trans_test = Trans_data[TrainingDataSize:TrainingDataSize+TestingDataSize, :]
    assert InputNum == Trans_train.shape[1]
else:
    data = scio.loadmat('data/ThinFilms/data_TF_100-300nm.mat')
    # data = scio.loadmat('data/ThinFilms/data_TF_0-150nm.mat')
    StartWL = 400
    EndWL = 701
    Resolution = 2
    WL = np.arange(StartWL, EndWL, Resolution)
    assert WL.size == np.array(data['WL']).size - 50
    InputNum = len(WL)
    Trans_train = torch.tensor(data['Trans_train'][10:161, 0:TrainingDataSize], device=device_data, dtype=dtype).t()
    Trans_test = torch.tensor(data['Trans_test'][10:161, 0:TestingDataSize], device=device_test, dtype=dtype).t()

del data

fnet = torch.load(fnet_path)
OutputNum = fnet.state_dict()['0.weight'].data.size(1)
fnet.to(device_train)
fnet.eval()

rnet = nn.Sequential(
    nn.Linear(InputNum, 2000),
    nn.BatchNorm1d(2000),
    nn.LeakyReLU(),
    nn.Linear(2000, 2000),
    nn.BatchNorm1d(2000),
    nn.LeakyReLU(),
    nn.Linear(2000, 800),
    nn.BatchNorm1d(800),
    nn.LeakyReLU(),
    nn.Linear(800, 800),
    nn.BatchNorm1d(800),
    nn.LeakyReLU(),
    nn.Linear(800, 100),
    nn.BatchNorm1d(100),
    nn.LeakyReLU(),
    nn.Linear(100, OutputNum),
    nn.Sigmoid(),
)
rnet.to(device_train)

LossFcn = nn.MSELoss(reduction='mean')
#optimizer = AdaBelief(rnet.parameters(), lr=lr, eps=1e-8, betas=(0.9, 0.999), weight_decouple=False, rectify=False)
optimizer = torch.optim.Adam(rnet.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

loss = torch.tensor([0], device=device_train)
loss_train = torch.zeros(math.ceil(EpochNum / TestInterval))
loss_test = torch.zeros(math.ceil(EpochNum / TestInterval))

os.makedirs(path, exist_ok=True)
log_file = open(path + 'TrainingLog.txt', 'w+')
time_start = time.time()
time_epoch0 = time_start
for epoch in range(EpochNum):
    Trans_train_shuffled = Trans_train[torch.randperm(TrainingDataSize), :]
    for i in range(0, TrainingDataSize // BatchSize):
        InputBatch = Trans_train_shuffled[i * BatchSize: i * BatchSize + BatchSize, :].to(device_train)
        Output_pred = fnet((params_max - params_min).to(device_train).float() * rnet(InputBatch) + params_min.to(device_train).float())
        loss = LossFcn(InputBatch, Output_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    if epoch % TestInterval == 0:
        fnet.to(device_test)
        rnet.to(device_test)
        rnet.eval()
        Out_test_params = (params_max - params_min).to(device_test).float() * rnet(Trans_test) + params_min.to(device_test).float()
        Out_test_pred = fnet(Out_test_params)
        fnet.to(device_train)
        rnet.to(device_train)
        rnet.train()
        loss_train[epoch // TestInterval] = loss.data
        loss_t = LossFcn(Trans_test, Out_test_pred)
        loss_test[epoch // TestInterval] = loss_t.data
        if epoch == 0:
            time_epoch0 = time.time()
            time_remain = (time_epoch0 - time_start) * EpochNum
        else:
            time_remain = (time.time() - time_epoch0) / epoch * (EpochNum - epoch)
        print('Epoch: ', epoch, '| train loss: %.5f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
              '| learn rate: %.8f' % scheduler.get_lr()[0], '| remaining time: %.0fs (to %s)'
              % (time_remain, time.strftime('%H:%M:%S', time.localtime(time.time() + time_remain))))
        print('Epoch: ', epoch, '| train loss: %.5f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
              '| learn rate: %.8f' % scheduler.get_lr()[0], file=log_file)
time_end = time.time()
time_total = time_end - time_start
m, s = divmod(time_total, 60)
h, m = divmod(m, 60)
print('Training time: %.0fs (%dh%02dm%02ds)' % (time_total, h, m, s))
print('Training time: %.0fs (%dh%02dm%02ds)' % (time_total, h, m, s), file=log_file)

rnet.eval()
torch.save(rnet, path + 'rnet.pkl')
fnet.to(device_test)
rnet.to(device_test)

Params_temp = (params_max - params_min).to(device_test).float() * rnet(Trans_train[0, :].to(device_test).unsqueeze(0)).squeeze(0) + params_min.to(device_test).float()
Output_temp = fnet(Params_temp.unsqueeze(0)).squeeze(0)
FigureTrainLoss = LossFcn(Trans_train[0, :].to(device_test), Output_temp)

print('Structure parameters of curve in figure \'train.png\':')
print('Structure parameters of curve in figure \'train.png\':', file=log_file)
print(Params_train[0, :])
print(Params_train[0, :], file=log_file)
print('Designed parameters of curve in figure \'train.png\':')
print('Designed parameters of curve in figure \'train.png\':', file=log_file)
print(Params_temp)
print(Params_temp, file=log_file)
print(Output_temp)
print(Output_temp, file=log_file)
plt.figure()
plt.plot(WL.T, Trans_train[0, :].cpu().numpy())
plt.plot(WL.T, Output_temp.detach().cpu().numpy())
plt.xlabel("Frequency(GHz)")
plt.ylabel("PCR")
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='lower right')
plt.savefig(path + 'train')
plt.show()

Params_temp = (params_max - params_min).to(device_test).float() * rnet(Trans_test[0, :].to(device_test).unsqueeze(0)).squeeze(0) + params_min.to(device_test).float()
Output_temp = fnet(Params_temp.unsqueeze(0)).squeeze(0)
FigureTestLoss = LossFcn(Trans_test[0, :].to(device_test), Output_temp)
np.set_printoptions(threshold=500)
print('Structure parameters of curve in figure \'test.png\':')
print('Structure parameters of curve in figure \'test.png\':', file=log_file)
print(Params_test[0, :])
print(Params_test[0, :], file=log_file)
print('Designed parameters of curve in figure \'test.png\':')
print('Designed parameters of curve in figure \'test.png\':', file=log_file)
print(Params_temp)
print(Params_temp, file=log_file)
print(Output_temp)
print(Output_temp, file=log_file)
plt.figure()
plt.plot(WL.T, Trans_test[0, :].cpu().numpy())
plt.plot(WL.T, Output_temp.detach().cpu().numpy())
plt.xlabel("Frequency(GHz)")
plt.ylabel("PCR")
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='lower right')
plt.savefig(path + 'test')
plt.show()


#scio.savemat(path +'pred_test.mat' , {'parameter':parameter , 'pcr': pcr})

print('Training finished!',
      '| loss in figure \'train.png\': %.5f' % FigureTrainLoss.data.item(),
      '| loss in figure \'test.png\': %.5f' % FigureTestLoss.data.item())
print('Training finished!',
      '| loss in figure \'train.png\': %.5f' % FigureTrainLoss.data.item(),
      '| loss in figure \'test.png\': %.5f' % FigureTestLoss.data.item(), file=log_file)
log_file.close()

plt.figure()
plt.plot(range(0, EpochNum, TestInterval), loss_train.detach().cpu().numpy())
plt.plot(range(0, EpochNum, TestInterval), loss_test.detach().cpu().numpy())
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.semilogy()
plt.legend(['Loss_train', 'Loss_test'], loc='upper right')
plt.savefig(path + 'loss')
plt.show()
