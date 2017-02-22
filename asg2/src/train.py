'''
Deep Learning Programming Assignment 1
--------------------------------------
Name: Agrawal Shubh Mohan
Roll No.: 14ME30003

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import random

def sigmoid(z):
    "sigmoid function"
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    "Derivative of sigmoid"
    return sigmoid(z)*(1-sigmoid(z))

class neuralNetwork(object):

    def __init__(self, layers, test_on = 0):
        '''Initilize network with weights. Any size can be defined'''
        self.num_layers = len(layers)
        self.layers = layers
        if test_on == 0:
            self.weights = [np.random.randn(y, x) for x,y in zip(layers[:-1], layers[1:])]
            self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        else:
            self.weights = np.load("weights/weights.npy")
            self.biases = np.load("weights/biases.npy")


    def feedforward(self, outputAct):
        ''' Feedforward the input activation while multiplying  with weight matrices'''        
        for b, w in zip(self.biases, self.weights):
            outputAct = sigmoid(np.dot(w, outputAct)+b)
        return outputAct


    def gradientDescent(self, all_data, epochs, mini_batch_size, lr, n_test=0):
        ''' Creates validation set, mini_batches and updates weights'''
        if n_test != 0 :
            test_data = all_data[:n_test]
        
        training_data = all_data[n_test:]    
        n_train = len(training_data)

        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [ training_data[k:k+mini_batch_size] for k in xrange(0, n_train, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)
            if n_test !=0:
                print "Epochs {0}: {1}%".format(j, self.evaluate(test_data))
            else:
                print "Epoch {0} complete".format(j)


    def update_mini_batch(self, mini_batch, lr):
        "Calculates the local error and derivative of loss"
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # vanila Update rule
        self.weights = [w-(lr/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(lr/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]



    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # feed it forward
        activation = x
        activations = [x]        # stores all the activations
        zs = []                  # stores all the z vectors
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # backward go
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp  # Local error
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        ''' Evaluates the validation data and gives accuracy'''        
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]   
        correct_labels = sum(int(x == y) for (x, y) in test_results)
        return (correct_labels*100.0)/len(test_data) 

    def cost_derivative(self, output_activations, y):
        ''' Mean square error loss fucntion derivative '''
        return (output_activations-y)

def dataProcess(trainX, trainY):
    
    # Convert to one hot vector
    labels = np.zeros((len(trainY), 10, 1))
    labels[np.arange(len(trainY)), trainY] = 1.0
     
    trainX = np.array(trainX).reshape(len(trainY), 784, 1).astype(np.float)
    trainX = trainX/255.0                               # Normalize
    training_data = zip(trainX, labels)
    return training_data

def train(trainX, trainY):
    ''' Trains the nerwork'''
    n_validation = 10000
    training_data = dataProcess(trainX, trainY)    
    
    #define net and training rule
    nNet = neuralNetwork([784, 50, 10])                           # 50 neurons in hidden layer # To fine tune the weights, add 1 as second arguement
    nNet.gradientDescent(training_data, 40, 10, 2.5, n_validation) # (epochs = 30, mini_batch_size = 10, Lr = 3.0)
    
    np.save("weights/weights.npy", nNet.weights)
    np.save("weights/biases.npy", nNet.biases)            

def test(testX):
    ''' testing the network in saved weights'''
    testX = np.array(testX).reshape(len(testX), 784, 1).astype(np.float)
    testX = testX/255.0                        #Normalize
    test_net = neuralNetwork([784, 50, 10], 1) #Define net   
    test_result = [np.argmax(test_net.feedforward(x)) for x in testX] #Feedforward  

    return test_result
