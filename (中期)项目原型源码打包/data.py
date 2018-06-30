import numpy as np
import pickle

file = open("D:/ChromeDownload/cifar-10-python/cifar-10-batches-py/data_batch_1", "rb")
dict = pickle.load(file,encoding='latin1')
print(dict['data'])
print(dict['labels'])
file.close()