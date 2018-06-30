import tensorflow as tf
import numpy as np
import pickle

def get_batch(image_W, image_H, batch_size, capacity):
    # image, label: 要生成batch的图像和标签list
    # image_W, image_H: 图片的宽高
    # batch_size: 每个batch有多少张图片
    #读取数据
    image = []
    label = []
    image_test = []
    label_test = []
    for i in range(1,6):
        file = open("D:/ChromeDownload/cifar-10-python/cifar-10-batches-py/data_batch_"+str(i), "rb")
        dict = pickle.load(file, encoding='latin1')
        image1 = dict['data']
        label1 = dict['labels']
        image1 = image1.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("float")
        image.append(image1)
        label.append(label1)
    #image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #image = tf.cast(image, tf.float32)
    image = np.concatenate(image)
    #image = tf.cast(image, tf.float32)
    label = np.concatenate(label)
    dict_test = pickle.load(open("D:/ChromeDownload/cifar-10-python/cifar-10-batches-py/test_batch", "rb"),encoding='latin1')
    image_test = dict_test['data']
    image_test = image_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    #image_test = tf.cast(image_test, tf.float32)
    #image_test = np.concatenate(image_test)
    label_test = dict_test['labels']
    return image, label, image_test, label_test


#依次获取batch
def getabatch(batch_size,batch_index,image,label):
    image_batch = []
    label_batch = []
    for i in range(batch_size):
        image_batch.append(image[i+batch_index*batch_size])
        label_batch.append(label[i+batch_index*batch_size])
    return image_batch,label_batch

