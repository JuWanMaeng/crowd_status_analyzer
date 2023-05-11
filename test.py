from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import torch
from MTL.model import resnet,MTL_model
import torchvision.transforms as transforms
from PIL import Image as im
import time

# labels
emo={0:'sad', 1:'happy', 2:'angry', 3:'disgust', 4:'surprise', 5:'fear', 6:'neutral'}
gender={0:'man',1:'woman'}
age={0:'youth',1:'student',2:'adult',3:'elder'}


# prepare model
device='cuda:0'
model=YOLO('ultralytics/models/v8/yolov8s.yaml')  # yolov8s model yaml file path
model=YOLO('weight/yolov8/s_best.pt')             # yolov8s weight

multi_model=MTL_model.MultiTaskModel(phase='test')  
wt=torch.load('weight/MTL/resnet18_5step_MTL_212.pt', map_location=torch.device('cuda:0'))
multi_model.load_state_dict(wt)
multi_model.to(device)

# start_time=time.time()
result=model(source='asian.png')  # source: frame(image) path or array
orig_img=result[0].orig_img       # original image
img=Image.fromarray(orig_img)
numpy_img=np.array(img)
opencv_img=cv2.cvtColor(numpy_img,cv2.COLOR_BGR2RGB)
boxes=result[0].boxes.xyxy.cpu().numpy()


if len(boxes)==0:        # 얼굴이 검출되지 않으면 inference 과정 생략
    print('no boxes')
else:
    orig_img=cv2.cvtColor(orig_img,cv2.COLOR_BGR2RGB)
    orig_faces=[]
    for box in boxes:
        x1=int(box[0])
        y1=int(box[1])
        x2=int(box[2])
        y2=int(box[3])
        
        cv2.rectangle(opencv_img,(x1,y1),(x2,y2),(255,0,0),2)  # opencv image에 사각형 그리기
        crop_img=orig_img[y1:y2,x1:x2,:]                       # original image를 crop 
        resized_img=cv2.resize(crop_img,(128,128))
        orig_faces.append(resized_img)                         # orig_faces list에 얼굴들 추가

    faces=np.array(orig_faces).astype(np.float32)/255.0
    faces=np.transpose(faces,(0,3,1,2))   # (B,H,W,C) -> (B,C,H,W)
    faces=torch.from_numpy(faces) 
    normalize=transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    faces=normalize(faces)

    faces=faces.to(device)
    outputs=multi_model(faces)
    gender_output,emo_output,age_output=outputs
    gender_pred=gender_output.argmax(1,keepdim=True)
    emo_pred=emo_output.argmax(1,keepdim=True)
    age_pred=age_output.argmax(1,keepdim=True)
    
# end_time=time.time()
#print(end_time-start_time)

# task별 분포를 측정하기 위한 딕셔너리
emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
age_dict={'youth':0,'student':0,'adult':0,'elder':0}
gender_dict={'man':0,'woman':0}

emo_labels=['sad','happy','angry','disgust','surprise','fear','neutral']
age_labels=['youth','student','adult','elder']
gender_labels=['man','woman']    

for i in range(len(orig_faces)):
    emo_dict[emo[emo_pred[i].item()]]+=1
    age_dict[age[age_pred[i].item()]]+=1
    gender_dict[gender[gender_pred[i].item()]]+=1
   
emo_ratio=list(emo_dict.values())
age_ratio=list(age_dict.values())
gender_ratio=list(gender_dict.values())


# 원형 그래프를 그리는 과정 만약 0이면 원형그래프에서 표시하지 않는다
fig1,ax1=plt.subplots()
wedges1, texts1, autotexts1 = ax1.pie(emo_ratio,labels=emo_labels,autopct=lambda x: '{:.1f}%'.format(x) if x > 0 else '',startangle=90)
# Remove label for 0% value
for i, text in enumerate(texts1):
    if emo_ratio[i] == 0:
        text.set_text('')   
             


fig2,ax2=plt.subplots()
wedges2, texts2, autotexts2 = ax2.pie(age_ratio,labels=age_labels,autopct=lambda x: '{:.1f}%'.format(x) if x > 0 else '',startangle=90)
# Remove label for 0% value
for i, text in enumerate(texts2):
    if age_ratio[i] == 0:
        text.set_text('')  
        
fig3,ax3=plt.subplots()
wedges3, texts3, autotexts3 = ax3.pie(gender_ratio,labels=gender_labels,autopct=lambda x: '{:.1f}%'.format(x) if x > 0 else '',startangle=90)
# Remove label for 0% value
for i, text in enumerate(texts3):
    if gender_ratio[i] == 0:
        text.set_text('')           

 
        
plt.show()
plt.imshow(opencv_img)
plt.show()