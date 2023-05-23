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

video_path = "asian_video.mp4"
cap = cv2.VideoCapture(video_path)
frame_count = 0
fps=0

if not cap.isOpened():
    print("Error opening video file")
    exit()
while True:
    start_time=time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    
    total_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
    youth_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
    student_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
    adult_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
    elder_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}

    man_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
    woman_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
    
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
        gender_output,emo_output,age_output=outputs
        gender_pred=gender_output.argmax(1,keepdim=True)
        emo_pred=emo_output.argmax(1,keepdim=True)
        age_pred=age_output.argmax(1,keepdim=True)
        results.append([gender_pred,emo_pred,age_pred])
        
    

    if frame_count%30==0 or frame_count==1:
        for i in range(len(orig_faces)):
            emotion=emo[emo_pred[i].item()]
            ages=age[age_pred[i].item()]
            genders=gender[gender_pred[i].item()]
            total_emo_dict[emo[emo_pred[i].item()]]+=1
            
            if ages == 'youth':
                youth_emo_dict[emotion]+=1
            elif ages=='student':
                student_emo_dict[emotion]+=1
            elif ages=='adult':
                adult_emo_dict[emotion]+=1
            else:
                elder_emo_dict[emotion]+=1
                
            if genders=='man':
                man_emo_dict[emotion]+=1
            else:
                woman_emo_dict[emotion]+=1
                
            
        total_emo_ratio=list(total_emo_dict.values())
        youth_emo_ratio=list(youth_emo_dict.values())
        student_emo_ratio=list(student_emo_dict.values())
        adult_emo_ratio=list(adult_emo_dict.values())
        elder_emo_ratio=list(elder_emo_dict.values())
        man_emo_ratio=list(man_emo_dict.values())
        woman_emo_ratio=list(woman_emo_dict.values())

        # Create subplots
        fig = plt.figure(figsize=(60,30))
        gs=gridspec.GridSpec(1,3)


        # ax1=plt.subplot(gs[0,:])
        # ax1.imshow(opencv_img)
        # ax1.axis('off')

        # Plot the pie chart
        ax2=plt.subplot(gs[0,0])
        wedges1, texts1, autotexts1 = ax2.pie(total_emo_ratio, labels=emo_labels, autopct=lambda x: '{:.1f}%'.format(x) if x > 0 else '', startangle=90)
        for i, text in enumerate(texts1):
            text.set_fontsize(50)
            if total_emo_ratio[i] == 0:
                text.set_text('')
        for autotext in autotexts1:
            autotext.set_fontsize(40)

        # Plot the line graph
        ax3 = plt.subplot(gs[0, 1])
        ax3.plot(emo_labels, youth_emo_ratio, label='Youth', linewidth=2.5)  # Increase the line thickness to 2.5
        ax3.plot(emo_labels, student_emo_ratio, label='Student', linewidth=2.5)  # Increase the line thickness to 2.5
        ax3.plot(emo_labels, adult_emo_ratio, label='Adult', linewidth=2.5)  # Increase the line thickness to 2.5
        ax3.plot(emo_labels, elder_emo_ratio, label='Elder', linewidth=2.5)  # Increase the line thickness to 2.5
        ax3.set_ylim(0, 6)
        ax3.set_xlabel('Emotions', fontsize=40)  # Increase the font size of the x-axis label
        ax3.set_ylabel('Number of Individuals', fontsize=40)  # Increase the font size of the y-axis label
        ax3.set_title('Emotional Distribution by Age', fontsize=40)  # Increase the font size of the title
        ax3.legend(fontsize=40)  # Increase the font size of the legend
        ax3.tick_params(axis='x', labelsize=40)
        ax3.tick_params(axis='y', labelsize=40)

        # Plot the bar graph
        ax4 = plt.subplot(gs[0, 2])
        x = range(len(emo_labels))
        ax4.bar(x, man_emo_ratio, width=0.4, align='center', label='Men')
        ax4.bar(x, woman_emo_ratio, width=0.4, align='edge', label='Women')
        ax4.set_xlabel('Emotions', fontsize=40)  # Increase the font size of the x-axis label
        ax4.set_ylabel('Number of Individuals', fontsize=40)  # Increase the font size of the y-axis label
        ax4.set_title('Emotion Distribution between Men and Women', fontsize=40)  # Increase the font size of the title
        ax4.set_xticks(x)
        ax4.set_xticklabels(emo_labels, fontsize=40)  # Increase the font size of the x-axis tick labels
        ax4.legend(fontsize=40)  # Increase the font size of the legend
        ax4.tick_params(axis='x', labelsize=40)
        ax4.tick_params(axis='y', labelsize=40)
        #print(f'graph time:{time.time()-start_time:4f}')
        
   

        # Adjust the layout
        plt.tight_layout()

        # Display the plot -> 이 부분에서 시간이 너무 오래 걸림
        plt.savefig('result/res.jpg')
        res=cv2.imread('result/res.jpg')
    
        cv2.namedWindow('Image 1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image 1',1600,1000)
        cv2.moveWindow('Image 1',1000,1)
        cv2.imshow('Image 1', res)
    
    cv2.namedWindow('open',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('open',800,600)
    cv2.moveWindow('open',100,300)
    cv2.imshow('open',opencv_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
    end_time=time.time()
    fps=1/(end_time-start_time)
      

cap.release()
