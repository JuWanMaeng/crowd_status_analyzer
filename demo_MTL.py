from ultralytics import YOLO
import matplotlib
#matplotlib.use('TKAgg')
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
from graph import generate_graph


emo={0:'sad', 1:'happy', 2:'angry', 3:'disgust', 4:'surprise', 5:'fear', 6:'neutral'}
gender={0:'man',1:'woman'}
age={0:'youth',1:'student',2:'adult',3:'elder'}

emo_labels=['sad','happy','angry','disgust','surprise','fear','neutral']
age_labels=['youth','student','adult','elder']
gender_labels=['man','woman']

device='cuda:0'
model=YOLO('ultralytics/models/v8/yolov8s.yaml')
model=YOLO('weight/yolov8/s_best.pt')

multi_model=MTL_model.MultiTaskModel(phase='test')
wt=torch.load('weight/MTL/resnet18_5step_MTL_212.pt', map_location=torch.device('cuda:0'))
multi_model.load_state_dict(wt)
multi_model.to(device)
multi_model.eval()

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
        outputs=multi_model(faces)
        emo_output=outputs[1]
        gender_output=outputs[0]
        age_output=outputs[2]
        
        gender_pred=gender_output.argmax(1,keepdim=True)
        emo_pred=emo_output.argmax(1,keepdim=True)
        age_pred=age_output.argmax(1,keepdim=True)

        
        graph_img = generate_graph(gender_pred,emo_pred,age_pred,len(orig_faces))

    
        cv2.namedWindow('Graph', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Graph',1500,900)
        cv2.moveWindow('Graph',1000,200)
        cv2.imshow('Graph', graph_img)
    
    cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video',900,600)
    cv2.moveWindow('Video',100,350)
    cv2.imshow('Video',opencv_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
    end_time=time.time()
    fps=1/(end_time-start_time)
      

cap.release()
print(f'{total_fps/frame_count:.2f}')