import tensorflow as tf
import numpy as np

import batch
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import neuralNetworkStructure as nns
import pylab


def evaluate():
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 10


        image_batch, label_batch, image_test, label_test = batch.get_batch(32,32,30,30)
        logit = nns.inference(image_batch, BATCH_SIZE, N_CLASSES)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[32, 32, 3])

        # you need to change the directories to yours.
        logs_train_dir = 'D:/ax/'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            # print("Reading checkpoints...")
            # ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            # if ckpt and ckpt.model_checkpoint_path:
            #     global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            #     saver.restore(sess, ckpt.model_checkpoint_path)
            #     print('Loading success, global_step is %s' % global_step)
            # else:
            #     print('No checkpoint file found')

            saver.restore(sess, 'D:/Ex/model.ckpt-4999')
            prediction = sess.run(logit, feed_dict={x: image_batch})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('This is a cellphone with possibility %.6f' % prediction[:, 0])
            elif max_index == 1:
                print('This is a bottle with possibility %.6f' % prediction[:, 1])
            else:
                print('This is a chair with possibility %.6f' % prediction[:, 2])


evaluate()

