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
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

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

def network_model(x_image):

    W1 = weight_variable([784, 100])
    b1 = bias_variable([100])

    y_1 = tf.matmul(x_image, W1) + b1

    W2 = weight_variable([100, 10])
    b2 = bias_variable([10])

    y_2 = tf.matmul(y_1, W2) + b2

    return y_2



def train(trainX, trainY):
    '''
    Complete this function.
    '''
    # convert to one hot 
    labels_array = np.array(trainY).astype(np.int)   
    labels_1h = np.zeros((len(labels_array), 10))
    labels_1h[np.arange(len(trainY)), labels_array] = 1.0
    trainY = labels_1h

    train_data, train_labels = normalize(np.array(trainX).reshape(-1, 784)), trainY

    print train_data[0]
    print train_labels[0]

    train_data, train_labels = shuffle(train_data, train_labels)
    train_data, train_labels, valid_data, valid_labels = separation(train_data, train_labels, 10000) # validation datasize is added here
    
    # reset the previous produced graph if any and start a new session
    tf.reset_default_graph()
    sess = tf.Session()

    x_image = tf.placeholder(tf.float32, [None, 784])
    target = tf.placeholder(tf.float32, [None, 10])

    output = network_model(x_image)
    prediction = tf.argmax(output,1)

    # store the operations
    tf.add_to_collection('train_dense', x_image)
    tf.add_to_collection('train_dense', prediction)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, target))
    train_step = tf.train.AdamOptimizer(3.00).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(target,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    minibatch = 10
    sess.run(tf.initialize_all_variables())

    # declare saver after producing graph and initialzing variables
    saver = tf.train.Saver()

    for epoch in range(30):
        train_data, train_labels = shuffle(train_data, train_labels)

        for k in xrange(0, len(train_labels), minibatch):   
            batch_input, batch_output = train_data[k : k + minibatch] , train_labels[k : k + minibatch] 
            sess.run(train_step, feed_dict = {x_image: batch_input, target: batch_output })

        valid_accuracy = sess.run(accuracy, feed_dict={x_image: valid_data, target: valid_labels})
        print "epoch %d, validation accuracy %g"%(epoch, valid_accuracy)


    saver.save(sess, "weights/dense/train_dense")
    sess.close()

    
def test(testX):
    
    testX = normalize(np.array(testX).reshape(-1, 784))

    tf.reset_default_graph()   # resets the graph, removes interference
    sess = tf.Session()

    saver = tf.train.import_meta_graph('weights/dense/train_dense.meta')  # restore graph
    saver.restore(sess, "weights/dense/train_dense")                     # restore weights

    x_image = tf.get_collection('train_dense')[0]                        # recapitulate operations
    prediction = tf.get_collection('train_dense')[1]
        
    testY = sess.run(prediction, feed_dict = {x_image: testX})

    sess.close()
    return testY
