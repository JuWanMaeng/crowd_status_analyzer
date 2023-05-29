import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel,QPushButton, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
import cv2
import torch
from MTL.model import resnet,MTL_model
import torchvision.transforms as transforms
from PIL import Image as im
import time

emo={0:'sad', 1:'happy', 2:'angry', 3:'disgust', 4:'surprise', 5:'fear', 6:'neutral'}
gender={0:'man',1:'woman'}
age={0:'youth',1:'student',2:'adult',3:'elder'}

emo_labels=['sad','happy','angry','disgust','surprise','fear','neutral']
age_labels=['youth','student','adult','elder']
gender_labels=['man','woman']




class GraphWindow(QMainWindow):
    def __init__(self,dicts):
        super().__init__()
        
        
        self.status=1
        self.dicts=dicts

        # Set up the main window
        self.setWindowTitle("Graph")
        self.setGeometry(1000,100, 1300, 1300)

        # Create the Matplotlib figure and canvas
        self.figure = Figure(figsize=(16,16))
        self.canvas = FigureCanvas(self.figure)

        self.figure2 = Figure(figsize=(16,16))
        self.canvas2 = FigureCanvas(self.figure2)
        
        self.figure3 = Figure(figsize=(16,16))
        self.canvas3 = FigureCanvas(self.figure3)

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.canvas2)
        layout.addWidget(self.canvas3)
        
        # Create a widget to hold the layout
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Plot the data
        if self.status==1:
            self.plot_data(self.dicts)
        else:
            self.plot_data2(self.dicts)


        # Create a QHBoxLayout for the buttons
        button_layout = QHBoxLayout()

        # Create the toggle buttons
        self.toggle_button1 = QPushButton("Age Distribution")
        button_layout.addWidget(self.toggle_button1)
        self.toggle_button2 = QPushButton("Total Distribution")
        button_layout.addWidget(self.toggle_button2)
        self.toggle_button3 = QPushButton("Gender Distribution")
        button_layout.addWidget(self.toggle_button3)

        # Connect the buttons' clicked signals to the corresponding methods
        self.toggle_button1.clicked.connect(self.toggle_graph1)
        self.toggle_button2.clicked.connect(self.toggle_graph2)
        self.toggle_button3.clicked.connect(self.toggle_graph3)

        # Add the button layout to the main layout
        layout.addLayout(button_layout)

        
    def update_data(self,dicts):
        self.dicts=dicts
        if self.status == 1:
            
            self.figure = Figure(figsize=(16,16))
            self.canvas = FigureCanvas(self.figure)

            self.figure2 = Figure(figsize=(16,16))
            self.canvas2 = FigureCanvas(self.figure2)
            
            self.figure3 = Figure(figsize=(16,16))
            self.canvas3 = FigureCanvas(self.figure3)

            # Set up the layout
            layout = QVBoxLayout()
            layout.addWidget(self.canvas)
            layout.addWidget(self.canvas2)
            layout.addWidget(self.canvas3)
            
            # Create a widget to hold the layout
            widget = QWidget()
            widget.setLayout(layout)
            self.setCentralWidget(widget)  
            
            # Create a QHBoxLayout for the buttons
            button_layout = QHBoxLayout()

            # Create the toggle buttons
            self.toggle_button1 = QPushButton("Age Distribution")
            button_layout.addWidget(self.toggle_button1)
            self.toggle_button2 = QPushButton("Total Distribution")
            button_layout.addWidget(self.toggle_button2)
            self.toggle_button3 = QPushButton("Gender Distribution")
            button_layout.addWidget(self.toggle_button3)

            # Connect the buttons' clicked signals to the corresponding methods
            self.toggle_button1.clicked.connect(self.toggle_graph1)
            self.toggle_button2.clicked.connect(self.toggle_graph2)
            self.toggle_button3.clicked.connect(self.toggle_graph3)

            # Add the button layout to the main layout
            layout.addLayout(button_layout)
            
              
            self.plot_data(self.dicts)
        elif self.status==2:
            self.figure = Figure(figsize=(16,16))
            self.canvas = FigureCanvas(self.figure)

            self.figure2 = Figure(figsize=(16,16))
            self.canvas2 = FigureCanvas(self.figure2)

            # Set up the layout
            layout = QVBoxLayout()
            layout.addWidget(self.canvas)
            layout.addWidget(self.canvas2)
            
            # Create a widget to hold the layout
            widget = QWidget()
            widget.setLayout(layout)
            self.setCentralWidget(widget)
            # Create a QHBoxLayout for the buttons
            button_layout = QHBoxLayout()

            # Create the toggle buttons
            self.toggle_button1 = QPushButton("Age Distribution")
            button_layout.addWidget(self.toggle_button1)
            self.toggle_button2 = QPushButton("Total Distribution")
            button_layout.addWidget(self.toggle_button2)
            self.toggle_button3 = QPushButton("Gender Distribution")
            button_layout.addWidget(self.toggle_button3)

            # Connect the buttons' clicked signals to the corresponding methods
            self.toggle_button1.clicked.connect(self.toggle_graph1)
            self.toggle_button2.clicked.connect(self.toggle_graph2)
            self.toggle_button3.clicked.connect(self.toggle_graph3)

            # Add the button layout to the main layout
            layout.addLayout(button_layout)        
            
            self.plot_data2(self.dicts)
            
        else:
            self.figure = Figure(figsize=(16,16))
            self.canvas = FigureCanvas(self.figure)
            layout = QVBoxLayout()
            layout.addWidget(self.canvas)
            
             # Create a widget to hold the layout
            widget = QWidget()
            widget.setLayout(layout)
            self.setCentralWidget(widget)
            # Create a QHBoxLayout for the buttons
            button_layout = QHBoxLayout()

            # Create the toggle buttons
            self.toggle_button1 = QPushButton("Age Distribution")
            button_layout.addWidget(self.toggle_button1)
            self.toggle_button2 = QPushButton("Total Distribution")
            button_layout.addWidget(self.toggle_button2)
            self.toggle_button3 = QPushButton("Gender Distribution")
            button_layout.addWidget(self.toggle_button3)   
            
            # Connect the buttons' clicked signals to the corresponding methods
            self.toggle_button1.clicked.connect(self.toggle_graph1)
            self.toggle_button2.clicked.connect(self.toggle_graph2)
            self.toggle_button3.clicked.connect(self.toggle_graph3)

            # Add the button layout to the main layout
            layout.addLayout(button_layout)    
            
            self.plot_data3(self.dicts)
                

    def plot_data(self, dicts):


    
        total_emo_ratio=list(dicts[0].values())
        youth_emo_ratio=list(dicts[1].values())
        student_emo_ratio=list(dicts[2].values())
        adult_emo_ratio=list(dicts[3].values())
        elder_emo_ratio=list(dicts[4].values())
        man_emo_ratio=list(dicts[5].values())
        woman_emo_ratio=list(dicts[6].values())
        
        # Clear any existing plots
        self.figure.clear()
        self.figure2.clear()
        self.figure3.clear()

        # Create axes for the plots
        ax1 = self.figure.add_subplot(111)
        ax2 = self.figure2.add_subplot(111)
        ax3 = self.figure3.add_subplot(111)

        index = np.arange(len(emo_labels))
        bar_width = 0.2

        ax1.bar(index, youth_emo_ratio, bar_width, label='Youth')
        ax1.bar(index + bar_width, student_emo_ratio, bar_width, label='Student')
        ax1.bar(index + 2 * bar_width, adult_emo_ratio, bar_width, label='Adult')
        ax1.bar(index + 3 * bar_width, elder_emo_ratio, bar_width, label='Elder')

        ax1.set_xlabel('Emotions', fontsize=10)
        ax1.set_ylabel('Number of Individuals', fontsize=10)
        ax1.set_title('Emotional Distribution by Age', fontsize=10)
        ax1.set_xticks(index + 1.5 * bar_width)
        ax1.set_xticklabels(emo_labels)
        ax1.legend(fontsize=10)

        wedges1, texts1, autotexts1 = ax2.pie(total_emo_ratio, labels=emo_labels,
                                              autopct=lambda x: '{:.1f}%'.format(x) if x > 0 else '', startangle=90)
        for i, text in enumerate(texts1):
            text.set_fontsize(10)
            if total_emo_ratio[i] == 0:
                text.set_text('')
        
        for autotext in autotexts1:
            autotext.set_fontsize(10)
        ax2.set_title('Total Emotion Distribution', fontsize=10)
        
        x = range(len(emo_labels))
        ax3.bar(x, man_emo_ratio, width=0.4, align='center', label='Men')
        ax3.bar(x, woman_emo_ratio, width=0.4, align='edge', label='Women')
        ax3.set_xlabel('Emotions', fontsize=10)  # Increase the font size of the x-axis label
        ax3.set_ylabel('Number of Individuals', fontsize=10)  # Increase the font size of the y-axis label
        ax3.set_title('Emotion Distribution between Men and Women', fontsize=10)  # Increase the font size of the title
        ax3.set_xticks(x)
        ax3.set_xticklabels(emo_labels, fontsize=10)  # Increase the font size of the x-axis tick labels
        ax3.legend(fontsize=10)  # Increase the font size of the legend
        ax3.tick_params(axis='x', labelsize=10)
        ax3.tick_params(axis='y', labelsize=10)
        
        # Store the axes in instance variables for later access
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3

        # Adjust subplot positions within the figure
        self.figure.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
        self.figure2.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
        self.figure3.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

        # Redraw the canvases
        self.canvas.draw()
        self.canvas2.draw()
        self.canvas3.draw()
        
    def plot_data2(self,dicts):
                
        youth_emo_ratio=list(dicts[1].values())
        student_emo_ratio=list(dicts[2].values())
        adult_emo_ratio=list(dicts[3].values())
        elder_emo_ratio=list(dicts[4].values())
        
        # Clear any existing plots
        self.figure.clear()
        self.figure2.clear()

        # Create axes for the plots
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        ax3 = self.figure2.add_subplot(121)
        ax4 = self.figure2.add_subplot(122)

        if sum(youth_emo_ratio) ==0:
            ax1.set_title('Youth Emotion Distribution (No Data)', fontsize=15)
        else:
            wedges1, texts1, autotexts1 = ax1.pie(youth_emo_ratio, labels=emo_labels,
                                                autopct=lambda x: '{:.1f}%'.format(x) if x > 0 else '', startangle=90)
            for i, text in enumerate(texts1):
                text.set_fontsize(20)
                if youth_emo_ratio[i] == 0:
                    text.set_text('')
            
            for autotext in autotexts1:
                autotext.set_fontsize(20)
            ax1.set_title('Youth Emotion Distribution', fontsize=15)
        
        if sum(student_emo_ratio) ==0:
            ax2.set_title('Student Emotion Distribution (No Data)', fontsize=15)
        else:
            wedges2, texts2, autotexts2 = ax2.pie(student_emo_ratio, labels=emo_labels,
                                                autopct=lambda x: '{:.1f}%'.format(x) if x > 0 else '', startangle=90)
            for i, text in enumerate(texts2):
                text.set_fontsize(20)
                if student_emo_ratio[i] == 0:
                    text.set_text('')
            
            for autotext in autotexts2:
                autotext.set_fontsize(20)
            ax2.set_title('Student Emotion Distribution', fontsize=15)
        
        if sum(adult_emo_ratio) ==0:
            ax3.set_title('Adult Emotion Distribution (No Data)', fontsize=15)
        else:
        
            wedges3, texts3, autotexts3 = ax3.pie(adult_emo_ratio, labels=emo_labels,
                                                autopct=lambda x: '{:.1f}%'.format(x) if x > 0 else '', startangle=90)
            for i, text in enumerate(texts3):
                text.set_fontsize(20)
                if adult_emo_ratio[i] == 0:
                    text.set_text('')
            
            for autotext in autotexts3:
                autotext.set_fontsize(20)
            ax3.set_title('Adult Emotion Distribution', fontsize=15)
        
        if sum(elder_emo_ratio) ==0:
            ax4.set_title('Elder Emotion Distribution (No Data)', fontsize=15)
        else:
            
            wedges4, texts4, autotexts4 = ax4.pie(elder_emo_ratio, labels=emo_labels,
                                                autopct=lambda x: '{:.1f}%'.format(x) if x > 0 else '', startangle=90)
            for i, text in enumerate(texts4):
                text.set_fontsize(20)
                if elder_emo_ratio[i] == 0:
                    text.set_text('')
            
            for autotext in autotexts4:
                autotext.set_fontsize(20)
            ax4.set_title('Elder Emotion Distribution', fontsize=15)
 
        self.figure.subplots_adjust(wspace=0.5)
        self.figure2.subplots_adjust(wspace=0.5)
        # Redraw the canvases
        self.canvas.draw()
        self.canvas2.draw()

    def plot_data3(self,dicts):
                
        man_emo_ratio=list(dicts[5].values())
        woman_emo_ratio=list(dicts[6].values())
        
        self.figure.clear()
        
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        
        if sum(man_emo_ratio) ==0:
            ax1.set_title('Men Emotion Distribution (No Data)', fontsize=15)
        else:
            wedges1, texts1, autotexts1 = ax1.pie(man_emo_ratio, labels=emo_labels,
                                                autopct=lambda x: '{:.1f}%'.format(x) if x > 0 else '', startangle=90)
            for i, text in enumerate(texts1):
                text.set_fontsize(20)
                if man_emo_ratio[i] == 0:
                    text.set_text('')
            
            for autotext in autotexts1:
                autotext.set_fontsize(20)
            ax1.set_title('Men Emotion Distribution', fontsize=15)
        
        if sum(woman_emo_ratio) ==0:
            ax2.set_title('Women Emotion Distribution (No Data)', fontsize=15)
        else:
            wedges2, texts2, autotexts2 = ax2.pie(woman_emo_ratio, labels=emo_labels,
                                                autopct=lambda x: '{:.1f}%'.format(x) if x > 0 else '', startangle=90)
            for i, text in enumerate(texts2):
                text.set_fontsize(20)
                if woman_emo_ratio[i] == 0:
                    text.set_text('')
            
            for autotext in autotexts2:
                autotext.set_fontsize(20)
            ax2.set_title('Women Emotion Distribution', fontsize=15)
            
        self.figure.subplots_adjust(wspace=0.5)
        self.canvas.draw()
        
    
    
    def toggle_graph1(self):
        self.status=2
        self.update_data(self.dicts)

            
    def toggle_graph2(self):
        self.status=1
        self.update_data(self.dicts)
        
    def toggle_graph3(self):
        self.status=3
        self.update_data(self.dicts)
        
    


