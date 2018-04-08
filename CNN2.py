# from IPython.display import Image
# Image('images/02_network_flowchart.png')
import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import prettytensor as pt
data=input_data.read_data_sets('data/MNIST/',one_hot=True)
filter_size1=5
num_filters1=16

filter_size2=5
num_filters2=36

fc_size=128
data.test.cls=np.argmax(data.test.labels,axis=1)
data.validation.cls=np.argmax(data.validation.labels,axis=1)
img_size = 28
img_size_flat=img_size*img_size
img_shape=(img_size,img_size)
num_channels=1
num_classes=10

combined_images=np.concatenate([data.train.images,data.validation.images])
combined_labels=np.concatenate([data.train.labels,data.validation.labels])
combined_size=len(combined_images)
train_size=int(0.8*combined_size)
validation_size=combined_size-train_size
def random_training_set():
    idx=np.random.permutation(combined_size)
    idx_train=idx[0:train_size]
    idx_validation=idx[train_size:]
    x_train = combined_images[idx_train,:]
    y_train = combined_labels[idx_train,:]
    x_validation = combined_images[idx_validation,:]
    y_validation = combined_labels[idx_validation,:]
    return x_train, y_train, x_validation, y_validation

def plot_images(images,cls_true,cls_pred=None):
    assert len(images) == len(cls_true) == 9

    fig,axes=plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    for i,ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape),cmap='binary')

        if cls_pred is None:
            xlabel = "true:{0}".format(cls_true[i])
        else:
            xlabel = "true:{0},pred:{1}".format(cls_true[i],cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
images=data.test.images[0:9]
cls_true=data.test.cls[0:9]
plot_images(images,cls_true=cls_true)

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05,shape=[length]))

#4_dim tensor
#(image num,x-axis of each image,y-axis of each image,channel of each image)


def new_conv_layer(input,num_input_channels,filter_size,num_filters,use_pooling=True):
    shape=[filter_size,filter_size,num_input_channels,num_filters]
    weights=new_weights(shape=shape)
    biases=new_biases(length=num_filters)
    layer=tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME')
    layer+=biases

    if use_pooling:
        layer=tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    layer=tf.nn.relu(layer)

    return layer,weights

def flatten_layer(layer):
    layer_shape=layer.get_shape()
    #(img_width,img_height,num_channel)
    num_features=np.array(layer_shape[1:4],dtype=int).prod()
    layer_flat=tf.reshape(layer,[-1,num_features])
    return layer_flat,num_features
keep_prob=tf.placeholder(tf.float32)
def new_fc_layer(input,num_inputs,num_outputs,use_relu=True,use_dropout=True):
    weights=new_weights(shape=[num_inputs,num_outputs])
    biases=new_biases(length=num_outputs)
    layer=tf.matmul(input,weights)+biases
    if use_relu:
        layer=tf.nn.relu(layer)
    if use_dropout:
        layer=tf.nn.dropout(layer,keep_prob=keep_prob)
    return layer

x=tf.placeholder(tf.float32,shape=[None,img_size_flat],name='x')
x_image=tf.reshape(x,[-1,img_size,img_size,num_channels])
y_true=tf.placeholder(tf.float32,shape=[None,10],name='y_true')
y_true_cls=tf.argmax(y_true,dimension=1)
#use pretty tensor
# x_pretty=pt.wrap(x_image)
# with pt.defaults_scope(activation_fn=tf.nn.relu):
#         y_pred,loss=x_pretty.\
#             conv2d(kernel=5,depth=16,name='layer_conv1').\
#             max_pool(kernel=2,stride=2).\
#             conv2d(kernel=5,depth=36,name='layer_conv2').\
#             max_pool(kernel=2,stride=2).\
#             flatten().\
#             fully_connected(size=128,name='layer_fc1').\
#             softmax_classifier(num_classes=10,labels=y_true)
#
# def get_weights_variable(layer_name):
#     with tf.variable_scope(layer_name,reuse=True):
#         variable=tf.get_variable('weights')
#     return variable
# weights_conv1=get_weights_variable(layer_name='layer_conv1')
layer_conv1,weights_conv1=new_conv_layer(input=x_image,num_input_channels=num_channels,filter_size=filter_size1,
                                         num_filters=num_filters1,use_pooling=True)
layer_conv2,weights_conv2=new_conv_layer(input=layer_conv1,num_input_channels=num_filters1,filter_size=filter_size2,
                                         num_filters=num_filters2,use_pooling=True)
layer_flat,num_features=flatten_layer(layer_conv2)
layer_fc1=new_fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=fc_size,use_relu=True,use_dropout=True)
layer_fc2=new_fc_layer(input=layer_fc1,num_inputs=fc_size,num_outputs=num_classes,use_relu=False,use_dropout=False)

y_pred=tf.nn.softmax(layer_fc2)
y_pred_cls=tf.argmax(y_pred,dimension=1)

cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)
cost=tf.reduce_mean(cross_entropy)
optimizer=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
#performance measure
correct_prediction=tf.equal(y_pred_cls,y_true_cls)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()
train_batch_size=64
total_iterations=0
best_validation_accuracy=0
last_improvement=0
require_improvement=1000
def random_batch(x_train,y_train):
    num_images = len(x_train)
    idx=np.random.choice(num_images,size=train_batch_size,replace=False)
    x_batch = x_train[idx,:]
    y_batch = y_train[idx,:]
    return x_batch, y_batch

def optimize(num_iterations,x_train,y_train):
    global total_iterations
    global best_validation_accuracy
    global last_improvement
    start_time=time.time()

    for i in range(total_iterations,total_iterations+num_iterations):
        x_batch,y_true_batch=random_batch(x_train,y_train)

        feed_dict_train={x:x_batch,y_true:y_true_batch,keep_prob:0.5}
        sess.run(optimizer,feed_dict=feed_dict_train)
        feed_dict_valid={x:data.validation.images,y_true:data.validation.labels,keep_prob:0.5}
    #     if i%100==0:
    #         acc=sess.run(accuracy,feed_dict=feed_dict_train)
    #         msg="optimization iteration: {0:>6},Training accuracy: {1}"
    #         print(msg.format(i+1,acc))
    # print("test accuracy %g"%sess.run(accuracy,feed_dict={x:data.test.images,y_true:data.test.labels,keep_prob:0.5}))
    # total_iterations+=num_iterations
    # end_time=time.time()
    # time_dif=end_time-start_time

        if (i%100 == 0):
            acc_train=sess.run(accuracy,feed_dict=feed_dict_train)
            acc_validation=sess.run(accuracy,feed_dict=feed_dict_valid)
            if acc_validation > best_validation_accuracy:
                best_validation_accuracy = acc_validation
                last_improvement = total_iterations
                saver.save(sess,"validation/best_validation.ckpt")
                improved_str='*'
            else:
                improved_str=''
            msg = "optimization iteration: {0:>6},Training accuracy: {1},validation accuracy:{2}"
            print(msg.format(i+1,acc_train,acc_validation))
        if total_iterations-last_improvement > require_improvement:
            print("no improvement found")
            break
def plot_example_errors(cls_pred,correct):
    incorrect=(correct==False)
    images=data.test.images[incorrect]
    cls_pred=cls_pred[incorrect]
    cls_true=data.test.cls[incorrect]
    plot_images(images=images[0:9],cls_true=cls_true[0:9],cls_pred=cls_pred[0:9])

def plot_confusion_matrix(cls_pred):
    # Get the true classifications for the test-set.
    cls_true = data.validation.cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks=np.arange(num_classes)
    plt.xticks(tick_marks,range(num_classes))
    plt.yticks(tick_marks,range(num_classes))

test_batch_size=256
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.validation.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.validation.images[i:j, :]

        # Get the associated labels.
        labels = data.validation.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels,keep_prob:0.5}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.validation.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)
num_networks=1
num_iterations=1100
if True:
    for i in range(num_networks):
        x_train,y_train,x_validation,y_validation = random_training_set()
        sess.run(tf.global_variables_initializer())
        optimize(num_iterations=num_iterations,
                 x_train=x_train,y_train=y_train)
        print()
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)
plt.show()


