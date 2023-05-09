import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms,models
import json
from PIL import Image as im
import cv2
import numpy as np
from torch.utils.data import DataLoader

import random
import numpy as np
import torch.utils.data as data
import glob


class EmotionDataset(data.Dataset): # 칼라 이미지 감정 dataset (100,100)


    def __init__(self,phase):
        super(EmotionDataset, self).__init__()
        
        self.phase=phase
        with open(f'data/Expw-F/{phase}.json', 'r') as f:
            
            self.data=json.load(f)
            
        mu, st = 0, 255
        self.train_transform= transforms.Compose([
            transforms.Resize((128,128)),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            transforms.RandomApply(
                [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        self.transform=transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            
        
        self.label={'sad':0, 'happy':1, 'angry':2, 'disgust':3, 'surprise':4, 'fear':5, 'neutral':6}
                

    def __getitem__(self, index):
        img_path,label=self.data[index]['img'],self.data[index]['label']
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=im.fromarray(img)
        
        label=self.label[label]
        
        
        if self.phase=='train':
            
            img=self.train_transform(img)
        else:
            img=self.transform(img)
        
        return img,label

    def __len__(self):
        return len(self.data)


class GenderDataset(data.Dataset):  # UTKFace 에서 성별만 따로 뽑음

    '''
    gender : 0,1
    
    '''
    def __init__(self,phase):
        super(GenderDataset, self).__init__()
        
        self.phase=phase
        with open(f'data/UTKFace/{phase}.json', 'r') as f:
            self.data=json.load(f)
            

        self.train_transform= transforms.Compose([
            transforms.Resize((128,128)),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            transforms.RandomApply(
                [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        self.transform=transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
        
        
        
        

    def __getitem__(self, index):
        img_path,gender=self.data[index]['img'],self.data[index]['label']
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=im.fromarray(img)
        
        if self.phase=='train':
            
            img=self.train_transform(img)
        else:
            img=self.transform(img)
        
        
        
        
        return img,gender

    def __len__(self):
        return len(self.data)

class AgeDataset(data.Dataset):  # UTKFace 에서 성별만 따로 뽑음

    '''
    age class 0,1,2,3  
    
    '''
    def __init__(self,phase):
        super(AgeDataset, self).__init__()
        
        self.phase=phase
        with open(f'data/face_age/{phase}.json', 'r') as f:
            self.data=json.load(f)
        
        self.train_transform= transforms.Compose([
            transforms.Resize((128,128)),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            transforms.RandomApply(
                [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        self.transform=transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
        self.label={'youth':0, 'student':1,'adult':2, 'elder':3}
        

    def __getitem__(self, index):
        img_path,age=self.data[index]['img'],self.data[index]['label']
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=im.fromarray(img)
        age=self.label[age]
        
        if self.phase=='train':
            
            img=self.train_transform(img)
        else:
            img=self.transform(img)
        return img,age

    def __len__(self):
        return len(self.data)






if __name__ == '__main__':
    tt=EmotionDataset(phase='val')
    train_dataloader=DataLoader(tt,batch_size=32,num_workers=8,shuffle=True)
    for i in range(len(train_dataloader.dataset)):
        print(tt[i][0].shape)
        break