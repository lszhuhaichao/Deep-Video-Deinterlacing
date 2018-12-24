import math
import numpy as np
import scipy
import tensorflow as tf
import random
import time
import os
import scipy.io
from math import log
from scipy.misc import imsave, imread, imresize
from datetime import datetime


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name)
def conv2d(x, W, strides=[1, 1, 1, 1], p='SAME', name=None):
    assert isinstance(x, tf.Tensor)
    return tf.nn.conv2d(x, W, strides=strides, padding=p, name=name)
def batch_norm(x):
    assert isinstance(x, tf.Tensor)
    mean, var = tf.nn.moments(x, axes=[1, 2, 3])
    return tf.nn.batch_normalization(x, mean, var, 0, 1, 1e-5)
def relu(x):
    assert isinstance(x, tf.Tensor)
    return tf.nn.relu(x)
def deconv2d(x, W, strides=[1, 1, 1, 1], p='SAME', name=None):
    assert isinstance(x, tf.Tensor)
    _, _, c, _ = W.get_shape().as_list()
    b, h, w, _ = x.get_shape().as_list()
    return tf.nn.conv2d_transpose(x, W, [b, strides[1]*h, strides[1]*w, c], strides=strides, padding=p, name=name)
def max_pool_2x2(x):
    assert isinstance(x, tf.Tensor)
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
def vupscale(x, upfield=True):

    sh = x.get_shape().as_list()
    out_size = [-1] +[sh[1]*2]+[s for s in sh[2:]]
    t = x
    if upfield:
        out = tf.concat([t, tf.zeros_like(t)],2)
    else:
        out = tf.concat([tf.zeros_like(t), t],2)
    out = tf.reshape(out,out_size)
    return out

def replaceField(x, input_image, upfield=True):

    upper_input = input_image[:,0::2,:,:]
    lower_input = input_image[:,1::2,:,:]

    if upfield:
        x = vupscale(x,upfield=False)
        upper_input = vupscale(upper_input,upfield=True)
        out = x + upper_input
    else:
        x = vupscale(x,upfield=True)
        lower_input = vupscale(lower_input,upfield=False)
        out = x + lower_input

    return out

def psnr(cost):
    return -10*log(cost)/log(10.0)

def batch_files(folder_path, image_files, batch_size):
    print("batch_files")
    nb_images = len(image_files)
    img = imread(folder_path+"/"+image_files[0], mode='RGB')
    print(folder_path+"/"+image_files[0])
    image_shape = img.shape
    for start_idx in range(0,nb_images,batch_size):
        current_batch_size = min(start_idx+batch_size,nb_images)-start_idx
        filesnames = image_files[start_idx:start_idx+current_batch_size]
        images = np.zeros((current_batch_size,) + image_shape)
        for i in range(0,current_batch_size):
            print(folder_path+"/"+image_files[start_idx+i])
            img = imread(folder_path+"/"+image_files[start_idx+i], mode='RGB')
            images[i] = img.astype('float32')/255.

        yield (filesnames, images)

class TSNet():

    def __init__(self):
        self.model_name = "TSNet"

    def createNet(self, input):

        #upper_Filed = input[:,0::2,:]
        #lower_Filed = input[:,1::2,:]
        #upper_Filed = vupscale(upper_Filed,upfield=True)
        #lower_Filed = vupscale(lower_Filed,upfield=False)
        self.c1 = weight_variable([3, 3, 1, 64], name='deinterlace/t_conv1_w')
        self.c2 = weight_variable([3, 3, 64, 64], name='deinterlace/t_conv2_w')
        self.c3 = weight_variable([1, 1, 64, 32], name='deinterlace/t_conv3_w')
        self.c41 = weight_variable([3, 3, 32, 32], name='deinterlace/t_conv41_w')
        self.c42 = weight_variable([3, 3, 32, 32], name='deinterlace/t_conv42_w')
        self.c51 = weight_variable([3, 3, 32, 1], name='deinterlace/t_conv51_w')
        self.c52 = weight_variable([3, 3, 32, 1], name='deinterlace/t_conv52_w')

        h = relu(conv2d(input, self.c1, name='t_conv1'))
        h = relu(conv2d(h, self.c2, name='t_conv2'))

        h = conv2d(h, self.c3, name='t_conv3')

        y = conv2d(h, self.c41, name='t_conv41')

        z = conv2d(h, self.c42, name='t_conv42')

        y = conv2d(y, self.c51, strides=[1, 2, 1, 1], name='t_conv51')
        y_full = replaceField(y, input, upfield=True)
        z = conv2d(z, self.c52, strides=[1, 2, 1, 1], name='t_conv52')
        z_full = replaceField(z, input, upfield=False)

        return (y, z, y_full, z_full)

    def deinterlace(self, args):
        img_path = args.img_path
        img = imread(img_path, mode='RGB')
        img = img.astype('float32') / 255.
        img_height, img_width, img_nchannels = img.shape

        input = np.swapaxes(np.swapaxes(img,0,2),1,2)
        input = input.reshape((3,img_height,img_width,1))
        im1 = np.zeros(img.shape).astype('float32')
        im2 = np.zeros(img.shape).astype('float32')
        im1[0::2,:,:] = img[0::2,:,:]
        im2[1::2,:,:] = img[1::2,:,:]
        if args.gpu > -1:
            device_ = '/gpu:{}'.format(args.gpu)
            print(device_)
        else:
            device_ = '/cpu:0'
        with tf.device(device_):
            x = tf.placeholder(tf.float32, shape=[3, img_height, img_width, 1])
            y,z,y_full, z_full = self.createNet(x)
            s_time = time.time()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            # Restore variables from disk.
            saver = tf.train.Saver()
            saver.restore(sess, args.model)
            print("Model restored.")
            lower, upper = sess.run([y,z], feed_dict={x: input})
            print('time: {} sec'.format(time.time() - s_time))
            lower_Field = np.swapaxes(np.swapaxes(lower,1,2),0,2)
            upper_Field = np.swapaxes(np.swapaxes(upper,1,2),0,2)
            im1[1::2,:,:] = lower_Field.reshape((int(img_height/2),img_width,3))
            im2[0::2,:,:] = upper_Field.reshape((int(img_height/2),img_width,3))
            im1 = im1.astype(np.float32)*255.0
            im1 = np.clip(im1, 0, 255).astype('uint8')
            im2 = im2.astype(np.float32)*255.0
            im2 = np.clip(im2, 0, 255).astype('uint8')
        input_filename = os.path.split(args.img_path)
        input_filename = os.path.splitext(input_filename[1])

        imsave("results/"+input_filename[0]+"_0" + input_filename[1], im1)
        imsave("results/"+input_filename[0]+"_1" + input_filename[1], im2)
