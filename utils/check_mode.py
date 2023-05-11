import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.datasets import GenderDataset,EmotionDataset,AgeDataset
import torch.optim.lr_scheduler as lr_scheduler

import os
import torch.nn as nn
import torch.optim as optim
import copy
import time
from MTL.model import resnet,mobilenet
from utils.train_utils import *
from utils.loss import FocalLoss

def check(mode,model):
    if mode==None:
        assert 'no mode selected!'
    else:  
        if mode=='emotion':
            # load dataset
            train_ds=EmotionDataset(phase='train')
            val_ds=EmotionDataset(phase='val')
            test_ds=EmotionDataset(phase='test')
            # make dataloade
            train_dl = DataLoader(train_ds, batch_size=256, shuffle=True,num_workers=8)
            val_dl = DataLoader(val_ds, batch_size=32, shuffle=True,num_workers=4)
            test_dl = DataLoader(test_ds, batch_size=16, shuffle=True,num_workers=1)
            
            print(mode)
            print(f'total train data:{len(train_dl.dataset)}')
            print(f'total val data:{len(val_dl.dataset)}')
            print(f'total test data:{len(test_dl.dataset)}')
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
            
        elif mode=='gender':
            # load dataset
            train_ds=GenderDataset(phase='train')
            val_ds=GenderDataset(phase='val')
            # make dataloade
            train_dl = DataLoader(train_ds, batch_size=256, shuffle=True,num_workers=16)
            val_dl = DataLoader(val_ds, batch_size=32, shuffle=True,num_workers=4)
            
            print(mode)
            print(f'total train data:{len(train_dl.dataset)}')
            print(f'total val data:{len(val_dl.dataset)}')
            
            criterion=nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        elif mode=='age':
            train_ds=AgeDataset(phase='train')
            val_ds=AgeDataset(phase='val')
            test_ds=AgeDataset(phase='test')
            
            # make dataloade
            train_dl = DataLoader(train_ds, batch_size=256, shuffle=True,num_workers=8)
            val_dl = DataLoader(val_ds, batch_size=32, shuffle=True,num_workers=4)
            test_dl=DataLoader(test_ds,batch_size=16,shuffle=True,num_workers=1)
            
            print(mode)
            print(f'total train data:{len(train_dl.dataset)}')
            print(f'total val data:{len(val_dl.dataset)}')
            print(f'total test data:{len(test_dl.dataset)}')
            
            criterion=nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            
            
        
        

            
    return train_dl,val_dl,test_dl,criterion,optimizer,scheduler