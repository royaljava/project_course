from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face
import cv2
import align.detect_face
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn.externals import joblib


frame_interval = 1
pathf = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(pathf)
cap = cv2.VideoCapture(0)


def image_resize(faces,frame,image_size,margin):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    img_list = []
    for (x, y, w, h) in faces:
        img = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)

def image_resize_no_align(faces,frame,image_size,margin):
    img_list = []
    for (x, y, w, h) in faces:
        img = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

def run():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model('model/20180408-102900')

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            c=0
            print("预加载结束")
            while True:
                ret, frame = cap.read()
                timeF = frame_interval
                if (c % timeF == 0):
                    print(11111111111111111111111111111111111111111)
                    find_results = []
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    if (len(faces)!=0):
                        images = image_resize_no_align(faces,frame,160,44)
                        print("yao lai dian tu zi ma")
                        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        model = joblib.load('knn_modified.model')
                        predict = model.predict(emb)
                        if(len(predict)>1):
                            for te in predict:
                                if(te == 1):
                                    find_results.append("Rong Yi")
                                if(te ==0):
                                    find_results.append("Wang Jiaping")
                                if(te==2):
                                    find_results.append("others")
                        else:
                            if(predict == 1):
                                find_results.append("Rong Yi")
                            if(predict==0):
                                find_results.append("Wang Jiaping")
                            if(predict==2):
                                find_results.append("others")
                        cv2.putText(frame, 'detected:{}'.format(find_results), (50, 100),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0),
                                    thickness=2, lineType=2)
                        print('识别结果为：{}'.format(predict))
                        print("GGGGGGGGGGGGGGGGGGG")
                c = c+1





                # 第一部分：抽帧（暂定一秒5帧）将检测的脸部存成image

                # 第二部分：在将img输入facenet并获得embedding向量
                # 第三部分：利用得到的embedding训练一个分类模型
                # 第四部分：进行脸部识别并绘制图现


                cv2.imshow('img', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()



    # Run forward pass to calculate embeddings

run()