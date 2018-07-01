import os
import numpy as np
import tensorflow as tf
# import input_data
import batch
import neuralNetworkStructure as nns


N_CLASSES = 10
IMG_W = 32  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 32
BATCH_SIZE = 50
CAPACITY = 2000
MAX_STEP = 10000  # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.001  # with current parameters, it is suggested to use learning rate<0.0001
logs_train_dir = 'D:/AAX'
checkpoint_path = 'D:/AAX/cifar10/baogao'

# %%
def run_training1():
    image_batch, label_batch, image_test, label_test = batch.get_batch(32, 32, BATCH_SIZE, 30)
    x = tf.placeholder(tf.float32, shape=[50, 32, 32, 3], name='x')
    y_ = tf.placeholder(tf.int32, shape=[50, ], name='y_')
    train_logits = nns.inference(x, BATCH_SIZE, N_CLASSES)
    train_loss = nns.losses(train_logits,y_)
    train_op = nns.trainning(train_loss, learning_rate)
    train__acc = nns.evaluation(train_logits, y_)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for epoch in range(200):
        for batch_index in range(1000):
            x_train,y_train = batch.getabatch(BATCH_SIZE,batch_index,image_batch,label_batch)
            _, err, ac = sess.run([train_op, train_loss, train__acc], feed_dict={x: x_train, y_: y_train})
        print('train accuracy:%d',ac)
    saver.save(sess, checkpoint_path)


    for test_index in range(200):
        x_test,y_test = batch.getabatch(BATCH_SIZE,test_index,image_test,label_test)
        _, error, acc = sess.run([train_op,train_loss,train__acc],feed_dict={x:x_test,y_:y_test})
        if(test_index % 50 == 0):
            print('test accuracy:%d',acc)
    sess.close()


run_training1()