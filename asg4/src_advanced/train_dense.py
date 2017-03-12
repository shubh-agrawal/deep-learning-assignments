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

def network_model(x_image):


    with tf.name_scope("hidden_1"):
        W1 = weight_variable([784, 100], "W1")
        b1 = bias_variable([100], "b1")
        y_1 = tf.matmul(x_image, W1) + b1

    with tf.name_scope("hidden_2"):
        W2 = weight_variable([100, 10], "W2")
        b2 = bias_variable([10], "b2")
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

    #print train_data[0]
    #print train_labels[0]

    train_data, train_labels = shuffle(train_data, train_labels)
    train_data, train_labels, valid_data, valid_labels = separation(train_data, train_labels, 10000) # validation datasize is added here
    
    # reset the previous produced graph if any and start a new session
    tf.reset_default_graph()
    sess = tf.Session()

    with tf.name_scope("images"):
        x_image = tf.placeholder(tf.float32, [None, 784], name="x-input")

    with tf.name_scope("labels"):
        target = tf.placeholder(tf.float32, [None, 10], name="y-output")

    with tf.name_scope("network_model"):
        output = network_model(x_image)

    with tf.name_scope("prediction"):
        prediction = tf.argmax(output,1)

    # store the operations
    tf.add_to_collection('train_dense', x_image)
    tf.add_to_collection('train_dense', prediction)


    with tf.name_scope("loss_function"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, target))

    with tf.name_scope("optimizer"):
        train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(target,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # create summar for these operations
    tf.scalar_summary("cost", cross_entropy)
    tf.scalar_summary("accuracy", accuracy)
    summary_op = tf.merge_all_summaries()

    sess.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter("/tmp/tensorflow/dense/1", graph=tf.get_default_graph())
    # declare saver after producing graph and initialzing variables
    saver = tf.train.Saver()

    minibatch = 10
    index = range(len(train_labels))
    for itr in range(150000):
        random_index_batch = np.random.choice(index, minibatch)       
        batch_input, batch_output = train_data[random_index_batch] , train_labels[random_index_batch] 

        _, summary = sess.run([train_step, summary_op], feed_dict = {x_image: batch_input, target: batch_output })
        writer.add_summary(summary, itr)

        if itr%500 == 0:
            valid_accuracy = sess.run(accuracy, feed_dict={x_image: valid_data, target: valid_labels})
            print "iteration %d, validation accuracy %g"%(itr, valid_accuracy)


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
