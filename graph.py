import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
import matplotlib.ticker as ticker
import concurrent.futures
import time
import matplotlib
matplotlib.use('agg')

emo={0:'sad', 1:'happy', 2:'angry', 3:'disgust', 4:'surprise', 5:'fear', 6:'neutral'}
gender={0:'man',1:'woman'}
age={0:'youth',1:'student',2:'adult',3:'elder'}

emo_labels=['sad','happy','angry','disgust','surprise','fear','neutral']
age_labels=['youth','student','adult','elder']
gender_labels=['man','woman']



def generate_graph(gender_pred,emo_pred,age_pred,length):

    start=time.time()
    total_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
    youth_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
    student_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
    adult_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
    elder_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}

    man_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
    woman_emo_dict={'sad':0,'happy':0,'angry':0,'disgust':0,'surprise':0,'fear':0,'neutral':0}
    
    for i in range(length):
        
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
    

    # Plot the line graph
    ax1 = plt.subplot(gs[0, 0])
    index = np.arange(len(emo_labels))
    bar_width = 0.2

    rects1 = ax1.bar(index, youth_emo_ratio, bar_width, label='Youth')
    rects2 = ax1.bar(index + bar_width, student_emo_ratio, bar_width, label='Student')
    rects3 = ax1.bar(index + 2 * bar_width, adult_emo_ratio, bar_width, label='Adult')
    rects4 = ax1.bar(index + 3 * bar_width, elder_emo_ratio, bar_width, label='Elder')

    ax1.set_xlabel('Emotions', fontsize=40)
    ax1.set_ylabel('Number of Individuals', fontsize=40)
    ax1.set_title('Emotional Distribution by Age', fontsize=40)
    ax1.set_xticks(index + 1.5 * bar_width)
    ax1.set_xticklabels(emo_labels)
    ax1.legend(fontsize=40)

    # Format y-axis tick labels as integers
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.tick_params(axis='x', labelsize=40)
    ax1.tick_params(axis='y', labelsize=40)


    # Plot the pie chart
    ax2=plt.subplot(gs[0,1])
    wedges1, texts1, autotexts1 = ax2.pie(total_emo_ratio, labels=emo_labels, autopct=lambda x: '{:.1f}%'.format(x) if x > 0 else '', startangle=90)
    for i, text in enumerate(texts1):
        text.set_fontsize(50)
        if total_emo_ratio[i] == 0:
            text.set_text('')
    for autotext in autotexts1:
        autotext.set_fontsize(40)
    ax2.set_title('Total Emotion Distribution',fontsize=50)

    # Plot the bar graph
    ax3 = plt.subplot(gs[0, 2])
    x = range(len(emo_labels))
    ax3.bar(x, man_emo_ratio, width=0.4, align='center', label='Men')
    ax3.bar(x, woman_emo_ratio, width=0.4, align='edge', label='Women')
    ax3.set_xlabel('Emotions', fontsize=40)  # Increase the font size of the x-axis label
    ax3.set_ylabel('Number of Individuals', fontsize=40)  # Increase the font size of the y-axis label
    ax3.set_title('Emotion Distribution between Men and Women', fontsize=40)  # Increase the font size of the title
    ax3.set_xticks(x)
    ax3.set_xticklabels(emo_labels, fontsize=40)  # Increase the font size of the x-axis tick labels
    ax3.legend(fontsize=40)  # Increase the font size of the legend
    ax3.tick_params(axis='x', labelsize=40)
    ax3.tick_params(axis='y', labelsize=40)
    #print(f'graph time:{time.time()-start_time:4f}')



    # Adjust the layout
    plt.tight_layout()
    print(f'first graph:{time.time()-start:.4f}')
    start=time.time()
  
    fig.canvas.draw()
    graph_img = np.array(fig.canvas.renderer.buffer_rgba())
    graph_img=cv2.cvtColor(graph_img,cv2.COLOR_BGR2RGB)
    print(f'graph time{time.time()-start:.4f}')
    
    return graph_img