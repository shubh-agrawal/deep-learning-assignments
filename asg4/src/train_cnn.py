'''
Deep Learning Programming Assignment 2
--------------------------------------
Name: Agrawal Shubh Mohan  
Roll No.: 14ME30003

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import tensorflow as tf

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def separation(shuffled_dataset, shuffled_labels, n_validate = 0):
    
    train_data = shuffled_dataset[n_validate:]
    train_labels = shuffled_labels[n_validate:]

    valid_data = shuffled_dataset[:n_validate]
    valid_labels = shuffled_labels[:n_validate]

    return train_data, train_labels, valid_data, valid_labels

def shuffle(train_data, train_labels):

    permutation = np.random.permutation(train_labels.shape[0])
    shuffled_dataset = train_data[permutation]
    shuffled_labels = train_labels[permutation]
    return shuffled_dataset, shuffled_labels

def normalize(data):
    normalized_data = (data.astype(np.float32) - 128.0) / 255.0
    return normalized_data    

def network_model(x_image, h_fc1_prob):  # layers are named as saver will then store them with given name as label. Easier to access specfic weights then.

    W_conv1 = weight_variable([5, 5, 1, 32], "W_conv1")
    b_conv1 = bias_variable([32], "b_conv1")

    h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME', name="h_conv1") + b_conv1
    h_relu1 = tf.nn.relu(h_conv1)
    h_pool1 = tf.nn.max_pool(h_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
    b_conv2 = bias_variable([64], "b_conv2")

    h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME', name="h_conv2") + b_conv2
    h_relu2 = tf.nn.relu(h_conv2)
    h_pool2 = tf.nn.max_pool(h_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    W_fc1 = weight_variable([7*7*64, 1024], "W_fc1")
    b_fc1 = bias_variable([1024], "b_fc1")
    
    h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    h_fc1_relu = tf.nn.relu(h_fc1)  
    h_fc1_drop = tf.nn.dropout(h_fc1_relu, h_fc1_prob)

    W_fc2 = weight_variable([1024, 10], "W_fc2")
    b_fc2 = bias_variable([10], "b_fc2")

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    return y_conv

def train(trainX, trainY):
    '''
    Complete this function.
    '''

    # Convert to one hot encoded
    labels_array = np.array(trainY).astype(np.int)  
    labels_1h = np.zeros((len(labels_array), 10))
    labels_1h[np.arange(len(trainY)), labels_array] = 1.0
    trainY = labels_1h

    train_data, train_labels = normalize(trainX), trainY

    train_data, train_labels = shuffle(train_data, train_labels)
    train_data, train_labels, valid_data, valid_labels = separation(train_data, train_labels, 10000) # validation datasize is added here

    tf.reset_default_graph()
    sess = tf.Session()
    
    x_image = tf.placeholder(tf.float32, [None, 28, 28, 1])
    target = tf.placeholder(tf.float32, [None, 10])

    h_fc1_prob = tf.placeholder(tf.float32)
    output = network_model(x_image, h_fc1_prob)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, target))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(target,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # only for serving / deploying purpose
    prediction = tf.argmax(output, 1)

    # store important operations for deploying
    tf.add_to_collection('train_cnn', x_image)
    tf.add_to_collection('train_cnn', h_fc1_prob)
    tf.add_to_collection('train_cnn', prediction)

    minibatch = 10
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    for epoch in range(30):
        train_data, train_labels = shuffle(train_data, train_labels)

        for k in xrange(0, len(train_labels), minibatch):   
            batch_input, batch_output = train_data[k : k + minibatch] , train_labels[k : k + minibatch] 
            sess.run(train_step, feed_dict = {x_image: batch_input, target: batch_output, h_fc1_prob: 0.60 })

        valid_accuracy = sess.run(accuracy, feed_dict={x_image: valid_data, target: valid_labels, h_fc1_prob: 1.0})
        print "epoch %d, validation accuracy %g"%(epoch, valid_accuracy)

    saver.save(sess, "weights/cnn/train_cnn")
    sess.close()

def test(testX):
    testX = normalize(np.array(testX).reshape(-1, 28, 28, 1))
    
    # reset graph if any graph was produced before this serving.
    tf.reset_default_graph()
    sess = tf.Session()

    saver = tf.train.import_meta_graph('weights/cnn/train_cnn.meta')
    saver.restore(sess, "weights/cnn/train_cnn")

    # recapitulate operations 
    x_image = tf.get_collection('train_cnn')[0]
    h_fc1_prob = tf.get_collection('train_cnn')[1]
    prediction = tf.get_collection('train_cnn')[2]
        
    testY = sess.run(prediction, feed_dict = {x_image: testX, h_fc1_prob: 1.0})
    sess.close()
    return testY
