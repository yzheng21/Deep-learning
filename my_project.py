import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
print('loading data')

def load_and_format(in_path):
    """ take the input data in .json format and return a df with the data and an np.array for the pictures """
    out_df = pd.read_json(in_path)
    out_images = out_df.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)
    out_images = np.stack(out_images).squeeze()
    return out_df, out_images


train_df, train_images = load_and_format('train.json')

test_df, test_images = load_and_format('test.json')

X_train, X_valid, y_train, y_valid = train_test_split(train_images,
                                                   train_df['is_iceberg'].as_matrix(),
                                                   test_size = 0.3
                                                   )
print('Train', X_train.shape, y_train.shape)
print('Validation', X_valid.shape, y_valid.shape)

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_valid = X_valid.astype(np.float32)
y_valid= y_valid.astype(np.float32)

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05,shape=[length]))

def new_conv_layer(input,num_input_channels,filter_size,num_filters,use_pooling=True):
    shape=[filter_size,filter_size,num_input_channels,num_filters]
    weights=new_weights(shape=shape)
    biases=new_biases(length=num_filters)
    layer=tf.nn.conv2d(input=input,filter=weights,strides=[1,3,3,1],padding='SAME')
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

def plot_confusion_matrix(cls_pred):

    cm = confusion_matrix(y_true=y_true_cls,
                          y_pred=cls_pred)
    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks=np.arange(num_classes)
    plt.xticks(tick_marks,range(num_classes))
    plt.yticks(tick_marks,range(num_classes))
    plt.show()

seed=42
np.random.seed(seed)


# Network Parameters
n_epochs =5
img_size=75
num_channels=2
num_classes = 2
dropout = 0.4
fc_size=128
filter_size1=5
num_filters1=16
filter_size2=5
num_filters2=36

X = tf.placeholder(tf.float32, shape=(None, img_size,img_size,num_channels), name="X")
x_image=tf.reshape(X,[-1,img_size,img_size,num_channels])
y = tf.placeholder(tf.int64, shape=(None,), name="y")
y_true_cls=tf.argmax(y,dimension=1)

layer_conv1,weights_conv1=new_conv_layer(input=x_image,num_input_channels=num_channels,filter_size=filter_size1,
                                         num_filters=num_filters1,use_pooling=True)
layer_conv2,weights_conv2=new_conv_layer(input=layer_conv1,num_input_channels=num_filters1,filter_size=filter_size2,
                                         num_filters=num_filters2,use_pooling=True)
layer_flat,num_features=flatten_layer(layer_conv2)
layer_fc1=new_fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=fc_size,use_relu=True,use_dropout=True)
layer_fc2=new_fc_layer(input=layer_fc1,num_inputs=fc_size,num_outputs=num_classes,use_relu=False,use_dropout=False)

y_pred=tf.nn.softmax(layer_fc2)
y_pred_cls=tf.argmax(y_pred,dimension=1)

cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=layer_fc2)
cost=tf.reduce_mean(cross_entropy)
optimizer=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
#performance measure
correct_prediction=tf.nn.in_top_k(layer_fc2, y, 1)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        sess.run(optimizer, feed_dict={X: X_train, y: y_train, keep_prob:0.4})
        acc_train = sess.run(accuracy,feed_dict={X: X_train, y: y_train, keep_prob:0.4})
        acc_test = sess.run(accuracy,feed_dict={X: X_valid,
                                        y: y_valid, keep_prob:0.4})

        if (epoch % 100 ==0):
            print(epoch, "Train accuracy:", acc_train, "Validation accuracy:", acc_test)

    # Z = layer_fc2.eval(feed_dict={X: test_images, keep_prob:0.4})
    # y_predict = Z[:,1]



# output = pd.DataFrame(test_df['id'])
# output['is_iceberg'] = y_predict
#
# output.to_csv('my_submission.csv', index=False)

sess.close()