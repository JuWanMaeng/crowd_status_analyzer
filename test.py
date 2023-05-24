from ultralytics import YOLO
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
import cv2
import torch
from MTL.model import resnet,MTL_model
import torchvision.transforms as transforms
from PIL import Image as imdddd
import time
import matplotlib.ticker as ticker
from torchvision.models import  resnet18
import torch.nn as nn
from graph import generate_graph
import threading
import queue

graph_queue=queue.Queue()

def generate_graph_async(gender_pred, emo_pred, age_pred, length):
    graph_img = generate_graph(gender_pred, emo_pred, age_pred, length)
    graph_queue.put(graph_img)



emo={0:'sad', 1:'happy', 2:'angry', 3:'disgust', 4:'surprise', 5:'fear', 6:'neutral'}
gender={0:'man',1:'woman'}
age={0:'youth',1:'student',2:'adult',3:'elder'}

emo_labels=['sad','happy','angry','disgust','surprise','fear','neutral']
age_labels=['youth','student','adult','elder']
gender_labels=['man','woman']

device='cuda:0'
model=YOLO('ultralytics/models/v8/yolov8s.yaml')
model=YOLO('weight/yolov8/s_best.pt')


emo_model=resnet.EmotionModel(phase='test')
emo_wt=torch.load('weight/classification/emotion128_resnet18.pt')
emo_model.load_state_dict(emo_wt)
emo_model.to(device)

age_model=resnet.AgeModel(phase='test')
age_wt=torch.load('weight/classification/age128_4class_merge_resnet18.pt')
age_model.load_state_dict(age_wt)
age_model.to(device)

gender_model=resnet18()
gender_model.fc=nn.Linear(512,2)
gender_wt=torch.load('weight/classification/UTK_gender_best_model.pt')
gender_model.load_state_dict(gender_wt)
gender_model.to(device)

emo_model.eval()
gender_model.eval()
age_model.eval()


video_path = "deadpool.mp4"
cap = cv2.VideoCapture(video_path)
frame_count = 0
total_fps=0
fps=0


if not cap.isOpened():
    print("Error opening video file")
    exit()
while True:
    total_fps+=fps
    start_time=time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1 
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results=[]

    result=model(image)
    orig_img=result[0].orig_img
    img=Image.fromarray(orig_img)
    numpy_img=np.array(img)
    opencv_img=cv2.cvtColor(numpy_img,cv2.COLOR_BGR2RGB)
    boxes=result[0].boxes.xyxy.cpu().numpy()

    if len(boxes)==0:
        print('no boxes')
    else:
        orig_img=cv2.cvtColor(orig_img,cv2.COLOR_BGR2RGB)
        orig_faces=[]
        for box in boxes:
            x1=int(box[0])
            y1=int(box[1])
            x2=int(box[2])
            y2=int(box[3])
            
            cv2.rectangle(opencv_img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(opencv_img, f'FPS: {fps:.0f}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0,255), 10)
            if frame_count%30==0 or frame_count==1:
                crop_img=orig_img[y1:y2,x1:x2,:]
                resized_img=cv2.resize(crop_img,(128,128))
                orig_faces.append(resized_img)
        
    
    cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video',900,600)
    cv2.moveWindow('Video',100,350)
    cv2.imshow('Video',opencv_img)
    if frame_count%30==0 or frame_count==1:
        start=time.time()
        
        crop_img=orig_img[y1:y2,x1:x2,:]
        resized_img=cv2.resize(crop_img,(128,128))
        orig_faces.append(resized_img)

        faces=np.array(orig_faces).astype(np.float32)/255.0
        faces=np.transpose(faces,(0,3,1,2))
        faces=torch.from_numpy(faces)
        normalize=transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        faces=normalize(faces)

        faces=faces.to(device)
        emo_output=emo_model(faces)
        gender_output=gender_model(faces)
        age_output=age_model(faces)
        
        gender_pred=gender_output.argmax(1,keepdim=True)
        emo_pred=emo_output.argmax(1,keepdim=True)
        age_pred=age_output.argmax(1,keepdim=True)
        print(f'inference time:{time.time()-start_time:.4f}')
        
        threading.Thread(target=generate_graph_async, args=(gender_pred, emo_pred, age_pred, len(orig_faces))).start()

        if not graph_queue.empty():
            graph_img=graph_queue.get()
            cv2.namedWindow('Graph', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Graph',1500,900)
            cv2.moveWindow('Graph',1000,200)
            cv2.imshow('Graph', graph_img)
            cv2.waitKey(1)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
    end_time=time.time()
    fps=1/(end_time-start_time)
      

cap.release()
print(f'{total_fps/frame_count:.2f}')