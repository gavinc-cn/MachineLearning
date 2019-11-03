import os
import pickle
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


CIFAR_DIR = "../data/cifar-10-batches-py"

print(os.listdir(CIFAR_DIR))

with open(os.path.join(CIFAR_DIR, 'data_batch_1'), 'rb') as f:
    # data = _pickle.load(f)
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    data = u.load()
    print(type(data))
    print(data.keys())
    print(type(data['data']))
    print(type(data['labels']))
    print(type(data['batch_label']))
    print(type(data['filenames']))
    print(data['data'].shape)
    print(data['data'][0:2])
    print(data['labels'][0:2])
    print(data['batch_label'])
    print(data['filenames'][0:2])

# 32 * 32 = 1024 * 3 = 3072
# RR-GG-BB

image_arr = data['data'][100]
print(image_arr)
image_arr = image_arr.reshape((3, 32, 32))
print(image_arr)
image_arr = image_arr.transpose((1, 2, 0))

imshow(image_arr)

