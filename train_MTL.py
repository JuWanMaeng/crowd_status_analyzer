import torch
import torch.nn as nn
from MTL.model.MTL_model import MultiTaskModel
from data.datasets import ExpDataset,GenderDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from tqdm import tqdm
from utils.train_utils import get_lr
from utils.loss import FocalLoss
import copy
import wandb,cv2,time
import numpy as np

from torchsummary import summary

###################################################################
wandb.init(project='emotoin+gender',entity='kookmin_ai')
device='cuda:1' if torch.cuda.is_available() else 'cpu'
backbone='resnet18'
model=MultiTaskModel(phase='train')
##################################################################
model_name=f'weight/MTL/{backbone}_only_FC.pt'
wandb.run.name=(f'{backbone}_01')
print(f'device:{device},backbone:{backbone}')

# build model
model=MultiTaskModel(phase='train')
model.to(device)

# Define the loss functions for each task

gender_criterion=nn.CrossEntropyLoss()
emotion_criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Define dataloader
train_gender_dataset=GenderDataset(phase='train')
train_emo_dataset=ExpDataset(phase='train')

val_gender_dataset=GenderDataset(phase='val')
val_emo_dataset=ExpDataset(phase='val')


train_gender_loader=DataLoader(train_gender_dataset,batch_size=256,shuffle=True,num_workers=8)
train_emo_loader=DataLoader(train_emo_dataset,batch_size=256,shuffle=True,num_workers=8)

val_gender_loader=DataLoader(val_gender_dataset,batch_size=64,shuffle=True,num_workers=2)
val_emo_loader=DataLoader(val_emo_dataset,batch_size=64,shuffle=True,num_workers=2)

print(f'total gender train data:{len(train_gender_loader.dataset)}')
print(f'total emotion train data:{len(train_emo_loader.dataset)}')

print(f'total gender val data:{len(val_gender_loader.dataset)}')
print(f'total emotion val data:{len(val_emo_loader.dataset)}')

'''
total gender train data:71168
total emotion train data:73437

total gender val data:23704
total emotion val data:18356
'''



# Define the weights for each task loss
emo_weight=1
gender_weight=1
num_epochs=1
best_loss=1000


for epoch in range(num_epochs):
    print('*'*60)
    current_lr=get_lr(optimizer)
    print(f'{epoch+1}/{num_epochs} current_lr:{current_lr}')
    
    train_total_loss=0
    train_gender_loss=0
    train_emotion_loss=0
    emo_corrects=0
    gender_corrects=0
    gender_tq=tqdm(train_gender_loader,ncols=80, smoothing=0, bar_format='train: {desc}|{bar}{r_bar}')
    
    model.train()
    for i, (gender_data, emo_data) in enumerate(zip(gender_tq, train_emo_loader)):
        

        gender_img,gender=gender_data
        emo_img,emotion=emo_data

        gender_img=gender_img.to(device)
        emo_img=emo_img.to(device)
        gender,emotion=gender.to(device),emotion.to(device)
        
        # set optimizer zero grad
        optimizer.zero_grad()

        # Forward pass
        emo_output=model(emo_img,task='E_P')
        emo_pred=torch.argmax(emo_output,dim=1)
        emo_corrects+=emo_pred.eq(emotion).sum().item()
        emo_loss=emotion_criterion(emo_output,emotion)
  
        gender_output=model(gender_img,task='A_G')
        gender_pred=torch.argmax(gender_output,dim=1)   # [64]
        gender_corrects+=gender_pred.eq(gender).sum().item()
        gender_loss=gender_criterion(gender_output,gender)
  
        # Compute the total loss
        common_loss = gender_weight * gender_loss+ emo_weight*emo_loss
        common_loss_value=common_loss.item()
        common_loss.backward(retain_graph=True)
        optimizer.step()
        
        # 각각의 total loss 저장
        train_total_loss+=common_loss_value
        train_gender_loss+=gender_loss.item()
        train_emotion_loss +=emo_loss.item()
    
    # epoch 당 각각의 loss
    train_loss = train_total_loss/len(train_gender_loader)
    train_gender_loss = train_gender_loss/len(train_gender_loader)
    train_emotoin_loss = train_emotion_loss/len(train_gender_loader)    
    # epoch 당 각각의 accuracy
    train_gender_accuracy=gender_corrects/len(train_gender_loader.dataset) * 100
    train_emotion_accuracy=emo_corrects/len(train_gender_loader.dataset) * 100
    
    # validation
    val_emo_tq=tqdm(val_emo_loader,ncols=80, smoothing=0, bar_format='val: {desc}|{bar}{r_bar}')
    total_loss=0
    val_gender_loss=0
    val_emotion_loss=0
    gender_corrects=0
    emo_corrects=0
    model.eval()
    with torch.no_grad():
        for i, (emo_data, gender_data) in enumerate(zip(val_emo_tq,val_gender_loader)):

            gender_img,gender=gender_data
            emo_img,emotion=emo_data

            gender_img=gender_img.to(device)
            emo_img=emo_img.to(device)
            gender,emotion=gender.to(device),emotion.to(device)
            
            
            # Forward pass
            gender_output=model(gender_img,task='A_G')
            gender_pred=torch.argmax(gender_output,dim=1)   # [64]
            gender_corrects+=gender_pred.eq(gender).sum().item()
            gender_loss=gender_criterion(gender_output,gender)

            emo_output=model(emo_img,task='E_P')
            emo_pred=torch.argmax(emo_output,dim=1)
            emo_loss=emotion_criterion(emo_output,emotion)
            emo_corrects+=emo_pred.eq(emotion).sum().item()
 
    
            # Compute the total loss
            loss = gender_loss + emo_loss
            total_loss+=loss.item()
            val_gender_loss+=gender_loss.item()
            val_emotion_loss+=emo_loss.item()
    
    
    val_loss=total_loss/len(val_emo_loader)
    val_gender_loss=val_gender_loss/len(val_emo_loader)
    val_emotion_loss=val_emotion_loss/len(val_emo_loader)
    val_gender_accuracy=gender_corrects/len(val_emo_loader.dataset) * 100
    val_emotion_accuracy=emo_corrects/len(val_emo_loader.dataset) * 100
    
    wandb.log({'train_total_loss':train_loss,
               'train_gender_loss':train_gender_loss,
               'train_emotion_loss':train_emotoin_loss,
                'val_total_loss':val_loss,
                'val_gender_loss':val_gender_loss,
                'val_emotoin_loss':val_emotion_loss,
                'train_gender_acc':train_gender_accuracy,
                'val_gender_acc':val_gender_accuracy,
                'train_emo_acc':train_emotion_accuracy,
                'val_emo_acc':val_emotion_accuracy},step=epoch+1)
    
    
    if val_loss<best_loss:
        best_loss=val_loss
        best_gender_acc=val_gender_accuracy
        best_emo_acc=val_emotion_accuracy
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(),model_name)
        print('Copied best model weights!')
        
    lr_scheduler.step(val_loss)
    if current_lr != get_lr(optimizer):
        print('Loading best model weights')
        model.load_state_dict(best_model_wts)
    
    
    
    print(f'train loss:{train_loss:.4f}, val loss:{val_loss:.4f}, gender_accuracy:{val_gender_accuracy:.2f}, emotion_accuracy:{val_emotion_accuracy:.2f} ')

print(f'best loss:{best_loss:.4f}, best_gender_acc:{best_gender_acc:.2f}, best_emo_acc:{best_emo_acc:.2f}')

wandb.log({'best gender acc':best_gender_acc,
           'best emotion acc':best_emo_acc})

example_images=[]
emo_label={0:'sad', 1:'happy', 2:'angry', 3:'disgust', 4:'surprise', 5:'fear', 6:'neutral'}
gender_label={0:'man',2:'woman'}
test_model=MultiTaskModel(phase='test')
test_model.load_state_dict(best_model_wts)
test_model.eval()
a=['ss.png','smile_man.png','angry_woman.png']
for image in a:
    with torch.no_grad():

        img=cv2.imread(image)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        img=cv2.resize(img,(128,128))
        img=np.transpose(img,(2,0,1))
        img=torch.FloatTensor(img).to(device)
        img=torch.unsqueeze(img,0)/255.0
        
        start=time.time()
        gender_output,emotion_output=model(img)
        infer_time=time.time()-start
        
        emo_pred=emotion_output.argmax(1,keepdim=True)
        gender_pred=gender_output.argmax(1,keepdim=True)
        emotion=emo_label[emo_pred.item()]
        gender=gender_label[gender_pred.item()]
    example_images.append(wandb.Image(
                    img, caption=f'Pred:{gender},{emotion}, inference_time:{infer_time:.4f}'))
wandb.log({"Image": example_images})