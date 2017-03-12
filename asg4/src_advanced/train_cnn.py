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

    with tf.name_scope("conv_1"):
        W_conv1 = weight_variable([5, 5, 1, 32], "W_conv1")
        b_conv1 = bias_variable([32], "b_conv1")
        h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME', name="h_conv1") + b_conv1
    
    with tf.name_scope("relu_1"):
        h_relu1 = tf.nn.relu(h_conv1)
    
    with tf.name_scope("pool_1"):
        h_pool1 = tf.nn.max_pool(h_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    
    with tf.name_scope("conv_2"): 
        W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
        b_conv2 = bias_variable([64], "b_conv2")
        h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME', name="h_conv2") + b_conv2
    

    with tf.name_scope("relu_2"):
        h_relu2 = tf.nn.relu(h_conv2)

    with tf.name_scope("pool_2"):    
        h_pool2 = tf.nn.max_pool(h_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    with tf.name_scope("flat_1"):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    with tf.name_scope("fc_1"):
        W_fc1 = weight_variable([7*7*64, 1024], "W_fc1")
        b_fc1 = bias_variable([1024], "b_fc1")
        h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1

    with tf.name_scope("fc_1_relu"):   
        h_fc1_relu = tf.nn.relu(h_fc1)  
    
    with tf.name_scope("fc_1_drop"):
        h_fc1_drop = tf.nn.dropout(h_fc1_relu, h_fc1_prob)


    with tf.name_scope("fc_2"):
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
    
    with tf.name_scope("images"):
        x_image = tf.placeholder(tf.float32, [None, 28, 28, 1], name="input")

    with tf.name_scope("labels"):    
        target = tf.placeholder(tf.float32, [None, 10], name="target")
    
    with tf.name_scope("network_model"):
        with tf.name_scope("fc_1_drop_param"):  # drop param must be part of network model
            h_fc1_prob = tf.placeholder(tf.float32)
        output = network_model(x_image, h_fc1_prob)

    with tf.name_scope("loss_function"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, target))
    
    with tf.name_scope("optimizer"):
        train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(target,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # only for serving / deploying purpose
    with tf.name_scope("test_prediction"):
        prediction = tf.argmax(output, 1)

    # store important operations for deploying
    tf.add_to_collection('train_cnn', x_image)
    tf.add_to_collection('train_cnn', h_fc1_prob)
    tf.add_to_collection('train_cnn', prediction)

    # collect summary of these operations
    tf.scalar_summary("loss", cross_entropy)
    tf.scalar_summary("accuracy", accuracy)
    summary_op = tf.merge_all_summaries()

    sess.run(tf.initialize_all_variables())

    writer = tf.train.SummaryWriter("/tmp/tensorflow/cnn/1", graph=tf.get_default_graph()) # for tensorboard
    saver = tf.train.Saver() # saving mechanism for graph and variables
    
    minibatch = 4
    index = range(len(train_labels))

    for itr in range(150000):
        random_index_batch = np.random.choice(index, minibatch)       
        batch_input, batch_output = train_data[random_index_batch] , train_labels[random_index_batch]
   
        _, summary = sess.run([train_step, summary_op], feed_dict = {x_image: batch_input, target: batch_output, h_fc1_prob: 0.60 })
        writer.add_summary(summary, itr) # append summary
    
        if itr%100 == 0:    
            valid_accuracy = sess.run(accuracy, feed_dict={x_image: valid_data[:1000], target: valid_labels[:1000], h_fc1_prob: 1.0})  # 1000 used since my computer cant handle 10000 at once. Future: use for loop to get accuracy batch wise and then take mean.
            print "iteration %d, validation accuracy %g"%(itr, valid_accuracy)

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
