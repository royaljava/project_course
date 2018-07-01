import sys
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication,QMessageBox)
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import pyqtSlot
from PyQt5.Qt import QLineEdit
import os
from sklearn import model_selection
from sklearn import metrics
from sklearn.externals import joblib

import cv2
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face
import restore

pathf = './haarcascade_frontalface_default.xml'
model_path ="./new_model"
face_cascade = cv2.CascadeClassifier(pathf)


class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        QToolTip.setFont(QFont('SansSerif', 10))
        self.setToolTip("This is a <b>QWidget</b> widget")

        btn_add = QPushButton("Add a Person", self)
        btn_add.setToolTip("Add a Person")
        btn_add.resize(btn_add.sizeHint())
        btn_add.move(50, 120)

        btn_identify = QPushButton("Entrance Guard", self)
        btn_identify.setToolTip("Entrance Guard")
        btn_identify.resize(btn_identify.sizeHint())
        btn_identify.move(50, 400)

        btn_sample = QPushButton("Sample", self)
        btn_sample.setToolTip("Sample the face")
        btn_sample.resize(btn_sample.sizeHint())
        btn_sample.move(50, 260)

        self.setGeometry(800, 800, 800, 600)
        self.setWindowTitle("Entrance Guard System")
        self.show()
        with tf.Graph().as_default():
            with tf.Session().as_default() as sess:
                # Load the model
                facenet.load_model('model/20180408-102900')
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.images_placeholder = images_placeholder
                self.embeddings = embeddings
                self.phase_train_placeholder = phase_train_placeholder
                self.sess=sess
        btn_add.clicked.connect(self.read)
        btn_identify.clicked.connect(self.detect)
        btn_sample.clicked.connect(self.sample)


    def read(self):
        file_name, ok = QFileDialog.getOpenFileNames(self, '多文件选择', './')
        if ok:
            if len(file_name)>9:
                local_path = os.path.dirname(os.path.dirname(file_name[0]))
                if os.path.exists(local_path+"/detect_face")!=0 :
                    if os.path.exists( local_path+"/embedding.txt")==0:
                        embedding_image(local_path,self.images_placeholder,self.embeddings,self.phase_train_placeholder,self.sess)
                        train_model(local_path)
                if os.path.exists(local_path+"/detect_face")==0:
                    detect_face(local_path,file_name)
                    embedding_image(local_path,self.images_placeholder,self.embeddings,self.phase_train_placeholder,self.sess)
                    #for roots,dirs,files in os.walk("./new_model"):
                        #print(dirs)
                    #print(os.path.dirname(file_name[0]))
                    train_model(os.path.dirname(file_name[0]))
                QMessageBox.information(self, "Message", "success",QMessageBox.Ok, QMessageBox.Ok)


    def detect(self):
        cap = cv2.VideoCapture(0)
        frame_time = 0
        print("开始识别")
        while True:
            ret, frame = cap.read()
            timeF = 1
            if (frame_time % timeF == 0):
                print(11111111111111111111111111111111111111111)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if (len(faces) != 0):
                    images = image_align(faces,frame,160,44)
                    cv2.imshow('img', frame)
                    cv2.waitKey(1)
                    if frame_time>80:
                        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
                        emb = self.sess.run(self.embeddings, feed_dict=feed_dict)
                        model_list = get_model_name(model_path)
                        access_flag = 0
                        for model_name in model_list:
                            model = joblib.load(model_name)
                            predict = model.predict(emb)
                            if (predict == 0):
                                print("Access")
                                access_flag = 1
                                break
                            if (predict == 1):
                                print("Denied")
                                continue
                        if access_flag ==0:
                            QMessageBox.information(self, "Message", "Denied", QMessageBox.Ok, QMessageBox.Ok)
                            break
                        else:
                            QMessageBox.information(self, "Message", "Access", QMessageBox.Ok, QMessageBox.Ok)
                            break
                        print("GGGGGGGGGGGGGGGGGGG")
            frame_time = frame_time + 1

            # 第一部分：抽帧（暂定一秒5帧）将检测的脸部存成image

            # 第二部分：在将img输入facenet并获得embedding向量
            # 第三部分：利用得到的embedding训练一个分类模型
            # 第四部分：进行脸部识别并绘制
        cap.release()
        cv2.destroyAllWindows()

    def sample(self):
        name = input("please input the name!")
        if os.path.exists(model_path+"/"+name)==0:
            os.mkdir(model_path+"/"+name)
            os.mkdir(model_path+"/"+name+"/detect_face")
            cap = cv2.VideoCapture(0)
            frame_count = 0
            pic_count = 0
            print("开始采样")
            while True:
                ret, frame = cap.read()
                timeF = 1
                #每10张采样一次
                sample_interval = 10
                if (frame_count % timeF == 0):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    if len(faces)!=0:
                        images = image_align(faces, frame, 160, 44)
                        # cv2.putText(frame, "please look at the cap", (50, 50),
                        #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0),
                        #             thickness=1.5, lineType=2)
                        cv2.imshow('img', frame)
                        cv2.waitKey(1)
                        if pic_count ==40:
                            break
                        if frame_count%sample_interval == 0:
                            pic_count = pic_count+1
                            pic_path = model_path+"/"+name+"/detect_face"+"/"+str(pic_count)+".jpg"
                            for (x, y, w, h) in faces:
                                cv2.imwrite(pic_path,frame[y:y+h,x:x+w])
                frame_count = frame_count +1
            cap.release()
            cv2.destroyAllWindows()



def embedding_image(path, images_placeholder, embeddings, phase_train_placeholder,sess):
    if os.path.exists(path + "/embedding.txt") == 0:
        path_detect = path + "/detect_face"
        face_path = get_file_name(path_detect)
        # print(face_path)
        images_align = restore.image_resize_no_align(face_path, image_size=160, margin=44)
        feed_dict = {images_placeholder: images_align, phase_train_placeholder: False}
        emb = sess.run(embeddings, feed_dict=feed_dict)
        print(emb)
        np.savetxt(path + "/embedding.txt", emb)


def train_model(file_path):
    other_path = os.path.dirname(file_path)+"/other/embedding.txt"
    sample_path = file_path+"/embedding.txt"
    print(other_path)
    print(sample_path)
    if os.path.exists(sample_path) != 0 :
        X = []
        f = open(sample_path)
        for line in f:
            list = line.strip('\n').split(' ')
            X.append(list)
        sample_length = len(X)
        f1 = open(other_path)
        for line in f1:
            list_other = line.strip('\n').split(' ')
            X.append(list_other)
        other_length = len(X)-sample_length
        train_y=np.zeros(len(X))
        for i in range(sample_length):
            train_y[i]=0
        for i in range(other_length):
            train_y[sample_length+i]=1
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, train_y, test_size=.2, random_state=42)

        # KNN Classifier
        def knn_classifier(train_x, train_y):
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier()
            model.fit(train_x, train_y)
            return model

        classifiers = knn_classifier
        model = classifiers(X_train, y_train)
        predict = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predict)
        print('accuracy: %.2f%%' % (100 * accuracy))
        # 保存模型
        joblib.dump(model, file_path+"/knn.model")


def detect_face(path,image_name):
    if os.path.exists(path+"/detect_face")==0:
        os.mkdir(path+"/detect_face")
        face_num = 0
        for file in image_name:
            img = cv2.imread(file)
            #print(file)
            #cv2.imshow('img', img)
            #cv2.waitKey(500)
            gray = cv2.imread(file, 0)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                continue
            for (x, y, w, h) in faces:
                cv2.imwrite(path+"/detect_face/" + str(face_num) + ".jpg", img[y:y + h, x:x + w])
                face_num = face_num + 1


def get_file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                 L.append(os.path.join(root, file))
    return L


def get_model_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.model':
                 L.append(os.path.join(root, file))
    return L


def image_align(faces,frame,image_size,margin):
    img_list = []
    for (x, y, w, h) in faces:
        img = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())