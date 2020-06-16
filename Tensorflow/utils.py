# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data:
import tensorflow as tf
import numpy as np
import cv2
def deconv2d(input, output_shape, kernal=(5, 5), strides=(2, 2), padding='SAME', activate_func=None, name='deconv2d'):
    if type(kernal) == list or type(kernal) == tuple:
        [k_h, k_w] = list(kernal)
    if type(strides) == list or type(strides) == tuple:
        [s_h, s_w] = list(strides)
    output_shape = list(output_shape)
    output_shape[0] = tf.shape(input)[0]
    with tf.variable_scope(name):
        w = tf.get_variable(name='weight', shape=[k_h, k_w, output_shape[-1], input.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=0.005))
        deconv = tf.nn.conv2d_transpose(input, filter=w, output_shape=tf.stack(output_shape, axis=0),
                                        strides=[1, s_h, s_w, 1], padding=padding)
        biases = tf.get_variable(name='biases', shape=[output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = deconv+ biases
        if activate_func:
            deconv = activate_func(deconv)
        return deconv

def show_in_one(path, images, column, row, show_size=[300, 300], blank_size=5):
    small_h, small_w = images[0].shape[:2]
    # column = int(show_size[1] / (small_w + blank_size))

    show_size[0] = small_h * row + row * blank_size
    show_size[1] = small_w * column + column * blank_size

    # row = int(show_size[0] / (small_h + blank_size))
    shape = [show_size[0], show_size[1]]
    for i in range(2, len(images[0].shape)):
        shape.append(images[0].shape[i])

    merge_img = np.zeros(tuple(shape), images[0].dtype)

    max_count = len(images)
    count = 0
    for i in range(row):
        if count >= max_count:
            break
        for j in range(column):
            if count < max_count:
                im = images[count]
                t_h_start = i * (small_h + blank_size)
                t_w_start = j * (small_w + blank_size)
                t_h_end = t_h_start + im.shape[0]
                t_w_end = t_w_start + im.shape[1]

                merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                count = count + 1
            else:
                break
    cv2.imwrite(path, merge_img)
    # cv2.namedWindow(window_name)
    # cv2.imshow(window_name, merge_img)

