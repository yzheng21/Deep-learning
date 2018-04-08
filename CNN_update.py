import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import prettytensor as pt
import os
import math
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
img_size_cropped=24
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

def plot_images(images,cls_true,cls_pred=None,smooth=True):
    assert len(images) == len(cls_true) == 9

    fig,axes=plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    for i,ax in enumerate(axes.flat):
        if smooth:
            interpolation='spline16'
        else:
            interpolation='nearest'
        ax.imshow(images[i].reshape(img_shape),cmap='binary')

        if cls_pred is None:
            xlabel = "true:{0}".format(cls_true[i])
        else:
            xlabel = "true:{0},pred:{1}".format(cls_true[i],cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# images=data.test.images[0:9]
# cls_true=data.test.cls[0:9]
# plot_images(images=images,cls_true=cls_true,smooth=False)

# #define placeholder
x=tf.placeholder(tf.float32,shape=[None,img_size_flat],name='x')
x_image=tf.reshape(x,[-1,img_size,img_size,num_channels])
y_true=tf.placeholder(tf.float32,shape=[None,10],name='y_true')
y_true_cls=tf.argmax(y_true,dimension=1)

def pre_process_image(image,training):
    if training:
        image = tf.random_crop(image, size=[img_size, img_size, num_channels])

        # image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_hue(image, max_delta=0.05)
        # image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        # image = tf.image.random_brightness(image, max_delta=0.2)
        # image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size,
                                                       target_width=img_size)
    return image


def pre_process(images,training):
    images=tf.map_fn(lambda image:pre_process_image(image,training),images)

    return images

distorted_images=pre_process(images=x_image,training=True)

def main_network(images,training):
    x_pretty=pt.wrap(images)
    if training:
        phase=pt.Phase.train
    else:
        phase=pt.Phase.infer

    with pt.defaults_scope(activation_fn=tf.nn.relu,phase=phase):
        y_pred,loss=x_pretty.\
                    conv2d(kernel=5,depth=16,name='layer_conv1').\
                    max_pool(kernel=2,stride=2).\
                    conv2d(kernel=5,depth=36,name='layer_conv2').\
                    max_pool(kernel=2,stride=2).\
                    flatten().\
                    fully_connected(size=128,name='layer_fc1').\
                    softmax_classifier(num_classes=10,labels=y_true)

    return y_pred,loss

def create_network(training):
    with tf.variable_scope('network',reuse=not training):
        images=x_image
        images=pre_process(images=images,training=training)
        y_pred,loss=main_network(images=images,training=training)

    return y_pred,loss
#for training phase
global_step=tf.Variable(initial_value=0,name='global_step',trainable=False)

_,loss=create_network(training=True)
optimizer=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss,global_step=global_step)

#for test phase
y_pred,_=create_network(training=False)
y_pred_cls=tf.argmax(y_pred,dimension=1)
correct_prediction=tf.equal(y_pred_cls,y_true_cls)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
saver=tf.train.Saver()

def get_weights_variable(layer_name):
    with tf.variable_scope("network/" + layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable

weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')

def get_layer_output(layer_name):
    # The name of the last operation of the convolutional layer.
    # This assumes you are using Relu as the activation-function.
    tensor_name = "network/" + layer_name + "/Relu:0"

    # Get the tensor with this name.
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

    return tensor

output_conv1 = get_layer_output(layer_name='layer_conv1')
output_conv2 = get_layer_output(layer_name='layer_conv2')

session=tf.Session()
save_dir='checkpoints/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'update_cnn')

try:
    print("Trying to restore last checkpoint ...")

    # Use TensorFlow to find the latest checkpoint - if any.
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

    # Try and load the data in the checkpoint.
    saver.restore(session, save_path=last_chk_path)

    # If we get to this point, the checkpoint was successfully loaded.
    print("Restored checkpoint from:", last_chk_path)
except:
    # If the above failed for some reason, simply
    # initialize all the variables for the TensorFlow graph.
    print("Failed to restore checkpoint. Initializing variables instead.")
    session.run(tf.global_variables_initializer())

train_batch_size=64

def random_batch(x_train, y_train):
    num_images = len(x_train)
    idx = np.random.choice(num_images, size=train_batch_size, replace=False)
    x_batch = x_train[idx, :]
    y_batch = y_train[idx, :]
    return x_batch, y_batch

def optimize(num_iterations,x_train,y_train):
    for i in range(num_iterations):
        x_batch,y_true_batch=random_batch(x_train, y_train)
        feed_dict_train={x:x_batch,y_true:y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)
        i_global,_=session.run([global_step,optimizer],
                                feed_dict=feed_dict_train)
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

            # Save a checkpoint to disk every 1000 iterations (and last).
        if (i_global % 1000 == 0) or (i == num_iterations - 1):
            # Save all variables of the TensorFlow graph to a
            # checkpoint. Append the global_step counter
            # to the filename so we save the last several checkpoints.
            saver.save(session,
                       save_path=save_path,
                       global_step=global_step)

            print("Saved checkpoint.")

def plot_example_error(cls_pred,correct):
    incorrect = (correct==False)
    images=data.test.images[incorrect]
    cls_pred=cls_pred[incorrect]
    cls_true=data.test.cls[incorrect]

def plot_confusion_matrix(cls_pred):
    cm=confusion_matrix(y_true=data.test.cls,
                        y_pred=cls_pred)
    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))

batch_size=256
def predict_cls(images,labels,cls_true):
    num_images=len(images)
    cls_pred=np.zeros(shape=num_images,dtype=np.int)
    i=0
    while i< num_images:
        j=min(i+batch_size,num_images)
        feed_dict={x:images[i:j,:],y_true:labels[i:j,:]}
        cls_pred[i:j]=session.run(y_pred_cls,feed_dict=feed_dict)
        i=j
    correct=(cls_true==cls_pred)
    return correct,cls_pred

def predict_cls_test():
    return predict_cls(images=data.test.images,
                      labels=data.test.labels,
                      cls_true=data.test.cls)

def classification_accuracy(correct):
    return correct.mean(),correct.sum()

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    correct,cls_pred=predict_cls_test()
    acc,num_correct=classification_accuracy(correct)
    num_images=len(correct)
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))
    if show_example_errors:
        print("Example errors:")
        plot_example_error(cls_pred=cls_pred, correct=correct)

        # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

def optimize_model():
    x_train, y_train,_,_=random_training_set()
    return optimize(num_iterations=1100,x_train=x_train,y_train=y_train)

def plot_conv_weights(weights,input_channel=0):
    w=session.run(weights)
    print("Min:  {0:.5f}, Max:   {1:.5f}".format(w.min(), w.max()))
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))
    w_min = np.min(w)
    w_max = np.max(w)
    abs_max = max(abs(w_min), abs(w_max))
    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=-abs_max, vmax=abs_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_layer_output(layer_output,image):
    feed_dict = {x: [image]}
    # Retrieve the output of the layer after inputting this image.
    values = session.run(layer_output, feed_dict=feed_dict)

    # Get the lowest and highest values.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    values_min = np.min(values)
    values_max = np.max(values)

    # Number of image channels output by the conv. layer.
    num_images = values.shape[3]

    # Number of grid-cells to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_images))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid image-channels.
        if i < num_images:
            # Get the images for the i'th output channel.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, vmin=values_min, vmax=values_max,
                      interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_distorted_image(image,cls_true):
    # Repeat the input image 9 times.
    image_duplicates = np.repeat(image[np.newaxis, :], 9, axis=0)

    # Create a feed-dict for TensorFlow.
    feed_dict = {x: image_duplicates}

    # Calculate only the pre-processing of the TensorFlow graph
    # which distorts the images in the feed-dict.
    result = session.run(distorted_images, feed_dict=feed_dict)

    # Plot the images.
    plot_images(images=result, cls_true=np.repeat(cls_true, 9))

def get_test_image(i):
    return data.test.images[i, :], data.test.cls[i]

img, cls = get_test_image(16)
plot_distorted_image(img, cls)
optimize_model()
print_test_accuracy(show_example_errors=True,show_confusion_matrix=True)
plot_conv_weights(weights=weights_conv1, input_channel=0)
plot_conv_weights(weights=weights_conv2, input_channel=1)
plot_layer_output(output_conv1, image=img)
plot_layer_output(output_conv2, image=img)
session.close()