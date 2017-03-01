import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# input and target output
x1 = tf.placeholder(tf.float32,shape=[None,784])
x2 = tf.placeholder(tf.float32,shape=[None,784])
x= tf.concat(1,[x1,x2])
#print x.get_shape()

y1_ = tf.placeholder(tf.float32,shape=[None,10])
y2_ = tf.placeholder(tf.float32,shape=[None,10])
y_ = tf.concat(1,[y1_,y2_])

# get the one-hot vectors of the sum
y1m = tf.argmax(y1_,1)
y2m = tf.argmax(y2_,1)
ym = tf.add(y1m,y2m)
y_ = tf.one_hot(ym,19)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def convpool_layer2d(x,widthx,widthy,features_in,features_out):
    W_conv = weight_variable([widthx,widthy,features_in,features_out])
    b_conv = bias_variable([features_out])
    h_conv = tf.nn.relu(conv2d(x,W_conv)+b_conv)
    h_pool = max_pool_2x2(h_conv);
    return h_pool

def fc_layer(x,nIn,nOut):
    W = weight_variable([nIn,nOut])
    b = bias_variable([nOut])
    return tf.nn.relu(tf.matmul(x,W)+b)

x_image = tf.reshape(x,[-1,28*2,28,1]);
# two convolutional layers
# first layer
h1 = convpool_layer2d(x_image,5,5,1,32)
# second layer
h2 = convpool_layer2d(h1,5,5,32,64)

h2_flat = tf.reshape(h2,[-1,2*7*7*64])
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
h_fc1 = fc_layer(h2_flat,2*7*7*64,2048)
# apply dropout (right now not doing this)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
# second layer
h_fc2 = fc_layer(h_fc1,2048,100);
# readout layer
n_out = 19
W_out = weight_variable([100,n_out])
b_out = bias_variable([n_out])

y_conv = tf.nn.sigmoid(tf.matmul(h_fc2, W_out)+b_out)

# the classical adder
# add the things up like a probability distribution


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#correct_prediction=tf.equal(tf.argmax(y_conv,1), tf.argmax(y1_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
accuracy = tf.reduce_mean(tf.square(tf.subtract(y_,y_conv)))
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# go
for i in range(10000):
    batch1 = mnist.train.next_batch(50)
    batch2 = mnist.train.next_batch(50)
    if i%100==0:
        train_accuracy = accuracy.eval(feed_dict={x1:batch1[0],x2:batch2[0],y1_:batch1[1],y2_:batch2[1],keep_prob:1.0})
        print("step %d, training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x1:batch1[0],x2:batch2[0],y1_:batch1[1],y2_:batch2[1],keep_prob:0.5})
    
#print("test accuracy %g"%accuracy.eval(feed_dict={x1_:mnist.test.images, y1_:mnist.test.labels,keep_prob:1.0}))
