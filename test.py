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
plt.switch_backend('TkAgg')
emo={0:'sad', 1:'happy', 2:'angry', 3:'disgust', 4:'surprise', 5:'fear', 6:'neutral'}
gender={0:'man',1:'woman'}
age={0:'youth',1:'student',2:'adult',3:'elder'}

total_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
youth_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
student_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
adult_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
elder_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}

man_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
woman_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}

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

results=[]
# start_time=time.time()
result=model(source='asian.png')
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
    
# end_time=time.time()


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
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(18,6))

# Plot the pie chart
wedges1, texts1, autotexts1 = ax1.pie(total_emo_ratio, labels=emo_labels, autopct=lambda x: '{:.1f}%'.format(x) if x > 0 else '', startangle=90)
for i, text in enumerate(texts1):
    if total_emo_ratio[i] == 0:
        text.set_text('')

# Plot the line graph
ax2.plot(emo_labels, youth_emo_ratio, label='Youth')
ax2.plot(emo_labels, student_emo_ratio, label='Student')
ax2.plot(emo_labels, adult_emo_ratio, label='Adult')
ax2.plot(emo_labels, elder_emo_ratio, label='Elder')
ax2.set_ylim(0, 24)
ax2.set_xlabel('Emotions')
ax2.set_ylabel('Emotion Ratio')
ax2.set_title('Emotional Distribution by Age')
ax2.legend()

# Set the x positions of the bars
x = range(len(emo_labels))

# Plotting
ax3.bar(x, man_emo_ratio, width=0.4, align='center', label='Men')
ax3.bar(x, woman_emo_ratio, width=0.4, align='edge', label='Women')

# Set the labels and title
ax3.set_xlabel('Emotions')
ax3.set_ylabel('Number of Individuals')
ax3.set_title('Emotion Distribution between Men and Women')
# Set the x-axis tick positions and labels
ax3.set_xticks(x, emo_labels)

# Add a legend
ax3.legend()

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()