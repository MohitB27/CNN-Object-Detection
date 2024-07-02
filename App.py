import tkinter as tk
from tkinter import simpledialog

import cv2
import cv2 as cv
import os
import PIL.Image, PIL.ImageTk

import Camera
import Model

path = r'C:\Users\mkbho\PycharmProjects\pythonProject1\dataset\Test'


class App:
    def __init__(self, window = tk.Tk(), window_title = "Camera Classifier" ):
        self.window = window
        self.window_title = window_title

        #keeps track of the images which will be named using this array
        self.counters = [1,1]

        self.model = Model.Model()

        #sets auto prediction to off by default
        self.auto_predict = False

        self.camera = Camera.Camera()

        self.init_gui()

        #Delay
        self.delay = 15
        #Update the tkinter gui
        self.update()

        #position of the window
        self.window.attributes('-topmost', True)
        self.window.mainloop()

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()

        self.btn_toggleauto = tk.Button(self.window, text="Auto Prediction", width=50, command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        self.Classname_one = simpledialog.askstring("Class one", "Enter the Name of the First Class : ", parent=self.window)

        self.Classname_two = simpledialog.askstring("Class two", "Enter the Name of the Second Class : ", parent=self.window)

        self.btn_class_one = tk.Button(self.window, text=self.Classname_one, width=50, command=lambda: self.save_for_class(1))
        self.btn_class_one.pack(anchor=tk.CENTER, expand=True)

        self.btn_class_two = tk.Button(self.window, text=self.Classname_two, width=50, command=lambda: self.save_for_class(2))
        self.btn_class_two.pack(anchor=tk.CENTER, expand=True)

        self.btn_train = tk.Button(self.window, text="Train Model", width=50, command=lambda: self.model.train_model(self.counters))
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(self.window, text="Predict", width=50, command=self.predict())
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text="Reset", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text="Class")
        self.class_label.config(font=("Ariel", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict

    #to save frames to folder
    def save_for_class(self, class_num):
        ret, frame = self.camera.get_Frame()
        if not os.path.exists('dataset/1'):
            os.mkdir('dataset/1')
        if not os.path.exists('dataset/2'):
            os.mkdir('dataset/2')

        cv.imwrite(f'dataset/{class_num}/frame{self.counters[class_num - 1]}.jpg', cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        img = PIL.Image.open( f'dataset/{class_num}/frame{self.counters[class_num - 1]}.jpg')
        img.thumbnail((150,150), PIL.Image.LANCZOS)
        img.save( f'dataset/{class_num}/frame{self.counters[class_num - 1]}.jpg')

        self.counters[class_num - 1] += 1

    def reset(self):
        for directory in ['1', '2']:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

        self.counters = [1, 1]
        self.model = Model.Model()
        self.class_label.config(text='CLASS')

    def update(self):
        if self.auto_predict:
            self.predict()

        ret, frame= self.camera.get_Frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def predict(self):
         frame = self.camera.get_Frame()
         prediction = self.model.predict(frame)

         if prediction is "Class 0":
              self.class_label.config(text=self.Classname_one)
              return self.Classname_one
         if prediction is "Class 1":
              self.class_label.config(text=self.Classname_two)
              return self.Classname_two