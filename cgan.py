from __future__ import print_function
import os, time, itertools
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

desired_size = 784
def lrelu(x, leak=0.2):
    return tf.nn.relu(tf.maximum(x, leak * x))

#conditional input, random noise, hidden layer num
def generator(x, z, layer_num = 2, isTrain=True, reuse=False):
    with tf.variable_scope("generator", reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()
        
        layers = [tf.concat([x,z], 1)]
        layers_size = [256]* layer_num
        for i in range(layer_num):
            temp_dense = tf.layers.dense(layers[-1], layers_size[i], kernel_initializer=w_init)
            temp_layer = lrelu(temp_dense)
            layers.append(temp_layer)
        
        out_dense = tf.layers.dense(layers[-1], desired_size, kernel_initializer=w_init)
        out_layer = lrelu(out_dense)
        return out_layer

def discriminator(x, y, layer_num = 2, isTrain=True, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()
        
        layers = [tf.concat([x,z], 1)]
        layers_size = [256] * layer_num
        for i in range(layer_num):
            temp_dense = tf.layers.dense(layers[-1], layers_size[i], kernel_initializer=w_init)
            temp_layer = lrelu(temp_dense)
            layer.append(temp_layer)
        out_dense = tf.layers.dense(layers[-1], 1, kernel_initializer=w_init)
        out_layer = tf.nn.sigmoid(out_dense)
        
        return out_layer, out_dense
