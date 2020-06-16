import numpy as np
import tensorflow as tf
import scipy.misc
import glob
import cv2

class DataSet():
    def __init__(self, num=2000, testnum=2000, img_size=64, batch_size=128, category='Fashion-Mnist'):
        self.img_size = img_size
        self.batch_size = batch_size
        self.num = num
        self.testnum = testnum
        self.category = category
        self.data, self.test_data = self.Load()

    def Load(self):
        if self.category == 'Mnist':
            print('Loading MNIST dataset ...')
            mnist = tf.keras.datasets.mnist
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
            data = train_images[:self.num]
            test_data = test_images[:self.testnum]

            img_size = self.img_size

            data = data.astype(np.float32)
            data = np.reshape(data, [-1, 28, 28, 1])

            test_data = test_data.astype(np.float32)
            test_data = np.reshape(test_data, [-1, 28, 28, 1])

            print(data.shape)
            data_resize = np.zeros([len(data), img_size, img_size, 1], dtype=np.float32)
            test_data_resize = np.zeros([len(test_data), img_size, img_size, 1], dtype=np.float32)
            for i in range(len(data)):
                data_resize[i, :, :, 0] = scipy.misc.imresize(np.squeeze(data[i, :, :, 0]) * 255.0,
                                                              [img_size, img_size])
            for i in range(len(test_data)):
                test_data_resize[i, :, :, 0] = scipy.misc.imresize(np.squeeze(test_data[i, :, :, 0]) * 255.0,
                                                                   [img_size, img_size])

            data = data_resize / 127.5 - 1
            test_data = test_data_resize / 127.5 - 1

        return data, test_data

    def NextBatch(self, index, test=False):
        if not test:
            return self.data[index * self.batch_size: (index + 1) * self.batch_size]
        else:
            return self.test_data[index * self.batch_size: (index + 1) * self.batch_size]

    def __len__(self):
        return self.num
