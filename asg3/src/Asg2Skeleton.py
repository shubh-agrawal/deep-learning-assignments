
# coding: utf-8

# Deep Learning Programming Assignment 2
# --------------------------------------
# Name: Agrawal Shubh Mohan 
# Roll No.: 14ME30003
# 
# Submission Instructions:
# 1. Fill your name and roll no in the space provided above.
# 2. Name your folder in format <Roll No>_<First Name>.
#     For example 12CS10001_Rohan
# 3. Submit a zipped format of the file (.zip only).
# 4. Submit all your codes. But do not submit any of your datafiles
# 5. From output files submit only the following 3 files. simOutput.csv, simSummary.csv, analogySolution.csv
# 6. Place the three files in a folder "output", inside the zip.

# In[59]:

import gzip
import os
import csv
import numpy as np
import itertools, random
import tensorflow as tf
from scipy.spatial.distance import cosine

## paths to files. Do not change this
simInputFile = "Q1/word-similarity-dataset"
analogyInputFile = "Q1/word-analogy-dataset"
vectorgzipFile = "Q1/glove.6B.300d.txt.gz"
vectorTxtFile = "Q1/glove.6B.300d.txt"   # If you extract and use the gz file, use this.
analogyTrainPath = "Q1/wordRep/"
simOutputFile = "Q1/simOutput.csv"
simSummaryFile = "Q1/simSummary.csv"
anaSoln = "Q1/analogySolution.csv"
Q4List = "Q4/wordList.csv"


# In[ ]:

# Similarity Dataset
simDataset = [item.split(" | ") for item in open(simInputFile).read().splitlines()] 
# creates a list of list
# Analogy dataset
analogyWordDataset = [[stuff.strip() for stuff in item.strip('\n').split('\n')] for item in open(analogyInputFile).read().split('\n\n')]

def vectorExtract(simD = simDataset, anaD = analogyWordDataset, vect = vectorgzipFile):
    simList = [stuff for item in simD for stuff in item]
    analogyList = [thing for item in anaD for stuff in item[0:6] for thing in stuff.split()]
    simList.extend(analogyList)
    wordList = set(simList)
    print len(wordList)
    wordDict = dict()
    
    vectorFile = gzip.open(vect, 'r')
    for line in vectorFile:
        if line.split()[0].strip() in wordList:
            wordDict[line.split()[0].strip()] = np.array(line.split()[1:]).astype(np.float32)
    
    
    vectorFile.close()
    print 'retrieved', len(wordDict.keys())
    return wordDict

# Extracting Vectors from Analogy and Similarity testing dataset
validateVectors = vectorExtract()
#print analogyDataset

# In[ ]:

# filtering analogyDataset only for questions with valid word vecs
tempList=[]
tempAnalogyDataset = []
options = ['a', 'b', 'c', 'd', 'e']
analogyLabels=[]
tempanalogyWordDataset = []
for block in analogyWordDataset:
	for line in block[:6]:
		wordPair = line.split()
		if wordPair[0] in validateVectors.keys() and wordPair[1] in validateVectors.keys():
			tempList.append(validateVectors[wordPair[0]] - validateVectors[wordPair[1]])
		else:
			tempList=[]
			break
	if len(tempList) != 0:			
		tempAnalogyDataset.append(tempList)
		analogyLabels.append(options.index(block[6]))
		tempanalogyWordDataset.append(block)
	tempList = []		

analogyLabels = np.array(analogyLabels).astype(np.int)
analogyDataset = tempAnalogyDataset
analogyWordDataset = tempanalogyWordDataset
print len(analogyDataset) 


# Dictionary of training pairs for the analogy task
trainDict = dict()
analogyWordList = []
for subDirs in os.listdir(analogyTrainPath):
    for files in os.listdir(analogyTrainPath+subDirs+'/'):
        f = open(analogyTrainPath+subDirs+'/'+files).read().splitlines()
        f_tuples = [x.split('\t') for x in f]
        trainDict[files] = f_tuples
        analogyWordList = analogyWordList + [word for pair in f_tuples for word in pair]
print len(trainDict.keys())
print len(analogyWordList)


if not os.path.exists(os.path.join(os.getcwd(), "analogyWordVecDict.npy")):
	analogyWordVecDict = dict()
	vectorFile = open(vectorTxtFile, 'r')
	for line in vectorFile:
		if line.split()[0].strip() in analogyWordList:
			analogyWordVecDict[line.split()[0].strip()] = line.split()[1:]
	print len(analogyWordVecDict.keys())
	np.save("analogyWordVecDict.npy", analogyWordVecDict)
else:
	analogyWordVecDict = np.load("analogyWordVecDict.npy").item()
	print len(analogyWordVecDict.keys())


if not os.path.exists(os.path.join(os.getcwd(), "trainWordVecDict.npy")):
	trainWordVecDict = dict()
	tempList = []
	for pairClass, pairs in zip(trainDict.keys(), trainDict.values()):
		for pair in pairs:
			if (pair[0] in analogyWordVecDict.keys()) and (pair[1] in analogyWordVecDict.keys()):
				#tempList.append([ [pair[0], analogyWordVecDict[pair[0]]], [pair[1], analogyWordVecDict[pair[1]]] ])
				tempList.append(np.array(analogyWordVecDict[pair[0]]).astype(np.float32) - np.array(analogyWordVecDict[pair[1]]).astype(np.float32))
		trainWordVecDict[pairClass] = tempList
		tempList = []
	print len(trainWordVecDict.keys())
	np.save("trainWordVecDict.npy", trainWordVecDict)
else:
	trainWordVecDict = np.load("trainWordVecDict.npy").item()
	print len(trainWordVecDict.keys())	
#print trainWordVecDict

positiveSamples = []
for pairClass in trainWordVecDict.keys():
	positiveSamples = positiveSamples + [x for x in itertools.combinations(trainWordVecDict[pairClass][:120], 2)]
positiveSamples = [ (np.append(x[0], x[1]), np.array([1.0, 0.0])) for x in positiveSamples]
print len(positiveSamples)
#print positiveSamples[5]

negativeSamples = []
for i in range(0, len(trainWordVecDict.keys()) - 1):
	negativeSamples = negativeSamples + list(itertools.product(trainWordVecDict.values()[i][:100], trainWordVecDict.values()[i+1][:100]))
negativeSamples = [ (np.append(x[0], x[1]), np.array([0.0, 1.0])) for x in negativeSamples]
print len(negativeSamples)
#print negativeSamples[5]

syntheticData = positiveSamples + negativeSamples
syntheticData = random.sample(syntheticData, len(syntheticData))
print len(syntheticData)


# In[58]:
def euclideanDistance(wordvec1, wordvec2):
	return np.linalg.norm(np.array(wordvec1).astype(np.float32) - np.array(wordvec2).astype(np.float32)) 

def manhattenDistance(wordvec1, wordvec2):
	dist = 0.0
	wordvec1 = np.array(wordvec1).astype(np.float32)
	wordvec2 = np.array(wordvec2).astype(np.float32)
	for i in range(len(wordvec2)):
		dist = dist + abs(wordvec1[i] - wordvec2[i])	
	return dist

def cosineSimilarity(wordvec1, wordvec2):
	wordvec1 = np.array(wordvec1).astype(np.float32)
	wordvec2 = np.array(wordvec2).astype(np.float32)
	return cosine(wordvec1, wordvec2)
		
def similarityTask(inputDS = simDataset, outputFile = simOutputFile, summaryFile=simSummaryFile, vectors=validateVectors):

	queWordList = [row[0] for row in simDataset]
	ansWordList = [row[1] for row in simDataset]
	choiceWordList = [row[1:] for row in simDataset]

	euclideanAnsList = []
	manhattenAnsList = []
	cosineAnsList = []

	euclideanMRR = 0.0
	manhattenMRR = 0.0
	cosineMRR = 0.0
	
	validCount = 0

	f1 = open(simOutputFile, "wb")
	csvwriter = csv.writer(f1, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)

	for index, word in enumerate(queWordList):

		if (set(choiceWordList[index]) < set(validateVectors.keys())) and (word in validateVectors.keys()):
			etempDistList = []
			ctempDistList = []
			mtempDistList = []
			for i in range(len(choiceWordList[index])):    					
				dist = euclideanDistance(validateVectors[word], validateVectors[choiceWordList[index][i]])
				etempDistList.append(dist)
				csvwriter.writerow([index, word, choiceWordList[index][i], 'E', dist])

				dist = manhattenDistance(validateVectors[word], validateVectors[choiceWordList[index][i]])
				mtempDistList.append(dist)
				csvwriter.writerow([index, word, choiceWordList[index][i], 'M', dist])

				dist = cosineSimilarity(validateVectors[word], validateVectors[choiceWordList[index][i]])
				ctempDistList.append(dist)
				csvwriter.writerow([index, word, choiceWordList[index][i], 'C', dist])	


			euclideanAnsList.append(choiceWordList[index][np.argmin(etempDistList)])
			manhattenAnsList.append(choiceWordList[index][np.argmin(mtempDistList)])
			cosineAnsList.append(choiceWordList[index][np.argmin(ctempDistList)])

			euclideanMRR = euclideanMRR + 1.0/(np.argmin(etempDistList) + 1.0)
			manhattenMRR = manhattenMRR + 1.0/(np.argmin(mtempDistList) + 1.0)
			cosineMRR = cosineMRR + 1.0/(np.argmin(ctempDistList) + 1.0)

			validCount = validCount + 1

		else:
			continue

	f1.close()		

	f2 = open(simSummaryFile, "wb")		
	csvwriter = csv.writer(f2, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)

	csvwriter.writerow(['E', len(set(euclideanAnsList) & set(ansWordList)), validCount, float(euclideanMRR/validCount)])
	csvwriter.writerow(['M', len(set(manhattenAnsList) & set(ansWordList)), validCount, float(manhattenMRR/validCount)])
	csvwriter.writerow(['C', len(set(cosineAnsList) & set(ansWordList)), validCount, float(cosineMRR/validCount)])
		
	f2.close()

	# remember to add MRR

# In[ ]:

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def separation(shuffled_dataset, n_validate = 0):
	
	trainData = shuffled_dataset[n_validate:]
	validData = shuffled_dataset[:n_validate]

	return trainData, validData	


def analogyTask(inputDS1=analogyDataset, inputDS2=analogyWordDataset, inputlabels = analogyLabels, outputFile = anaSoln, dataset= syntheticData ): # add more arguments if required



	#trainData, validData = separation(dataset, 20000)

	sess = tf.InteractiveSession()

	x_image = tf.placeholder(tf.float32, [None, 600])
	y_target = tf.placeholder(tf.float32, [None, 2])

	w1 = weight_variable([600, 200])
	b1 = bias_variable([200])

	
	h1 = tf.matmul(x_image, w1) + b1
	
	w2 = weight_variable([200, 2])
	b2 = bias_variable([2])

	y_output = tf.matmul(h1, w2) + b2

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_output, y_target))
	train_step = tf.train.AdamOptimizer(0.8).minimize(loss)
	
	correct_prediction = tf.equal(tf.argmax(y_output,1), tf.argmax(y_target,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	test_output = tf.nn.softmax(y_output)

	minibatch = 1000
	crossValidAcc = 0.0
	sess.run(tf.initialize_all_variables())


	print "5 Fold cross validation"

	for i in range(0, len(syntheticData) - len(syntheticData)%5, len(syntheticData)/5):
		validData = syntheticData[i: i + len(syntheticData)/5]
		trainData = syntheticData[0:i] + syntheticData[i + len(syntheticData)/5: len(syntheticData)]

		for epoch in range(10):
			trainData = random.sample(trainData, len(trainData))

			for k in xrange(0, len(trainData), minibatch):	
				batch_input, batch_output = [ x[0] for x in trainData[k: k + minibatch] ], [ x[1] for x in trainData[k : k + minibatch] ] 
				sess.run(train_step, feed_dict = {x_image: batch_input, y_target: batch_output })


			valid_input, valid_output = [ x[0] for x in validData ], [ x[1] for x in validData ]
			valid_accuracy = sess.run(accuracy, feed_dict={x_image: valid_input, y_target: valid_output})
			print "epoch %d, validation accuracy %g"%(epoch, valid_accuracy)

		crossValidAcc = crossValidAcc + sess.run(accuracy, feed_dict={x_image: valid_input, y_target: valid_output})
		

	#Test graph

	blockInput = []
	choiceOutput = []
	for i in range(0, len(analogyDataset)):
		for j in range(1, 6):
			blockInput.append(np.append(analogyDataset[i][0], analogyDataset[i][j]))		

		print sess.run(y_output, feed_dict={x_image: blockInput})	
		probPositive = [x[0] for x in sess.run(y_output, feed_dict={x_image: blockInput})]
		choiceOutput.append(np.argmax(probPositive))
		blockInput = []

	print choiceOutput
	correct_test = np.equal(np.array(choiceOutput).astype(np.int), np.array(analogyLabels).astype(np.int))
	test_accuracy = np.mean(correct_test)
	print test_accuracy

	f = open(anaSoln, "wb")
	csvwriter = csv.writer(f, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
	
	for i in range(len(analogyWordDataset)):
		csvwriter.writerow([analogyWordDataset[i][0], analogyWordDataset[i][analogyLabels[i] + 1], analogyWordDataset[i][choiceOutput[i] + 1] ])
	f.close()	



	"""
	Output a file, analogySolution.csv with the following entris
	Query word pair, Correct option, predicted option    
	"""

	print "Accuracy after 5 fold cross validation: ", crossValidAcc/5.0 
	return crossValidAcc/5.0 #valid_accuracy #return the accuracy of your model after 5 fold cross validation



# In[60]:

def derivedWOrdTask(inputFile = Q4List):
    print 'hello world'
    
    """
    Output vectors of 3 files:
    1)AnsFastText.txt - fastText vectors of derived words in wordList.csv
    2)AnsLzaridou.txt - Lazaridou vectors of the derived words in wordList.csv
    3)AnsModel.txt - Vectors for derived words as provided by the model
    
    For all the three files, each line should contain a derived word and its vector, exactly like 
    the format followed in "glove.6B.300d.txt"
    
    word<space>dim1<space>dim2........<space>dimN
    charitably 256.238 0.875 ...... 1.234
    
    """
    
    """
    The function should return 2 values
    1) Averaged cosine similarity between the corresponding words from output files 1 and 3, as well as 2 and 3.
    
        - if there are 3 derived words in wordList.csv, say word1, word2, word3
        then find the cosine similiryt between word1 in AnsFastText.txt and word1 in AnsModel.txt.
        - Repeat the same for word2 and word3.
        - Average the 3 cosine similarity values
        - DO the same for word1 to word3 between the files AnsLzaridou.txt and AnsModel.txt 
        and average the cosine simialities for valuse so obtained
        
    """
    return cosVal1,cosVal2
    


# In[ ]:

def main():
	similarityTask()
	anaSim = analogyTask()
	#derCos1,derCos2 = derivedWordTask
	print "Hello"

if __name__ == '__main__':
	main()