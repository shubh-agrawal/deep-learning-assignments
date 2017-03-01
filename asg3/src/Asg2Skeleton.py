
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
analogyDataset = [[stuff.strip() for stuff in item.strip('\n').split('\n')] for item in open(analogyInputFile).read().split('\n\n')]

def vectorExtract(simD = simDataset, anaD = analogyDataset, vect = vectorgzipFile):
    simList = [stuff for item in simD for stuff in item]
    analogyList = [thing for item in anaD for stuff in item[0:4] for thing in stuff.split()]
    simList.extend(analogyList)
    wordList = set(simList)
    print len(wordList)
    wordDict = dict()
    
    vectorFile = gzip.open(vect, 'r')
    for line in vectorFile:
        if line.split()[0].strip() in wordList:
            wordDict[line.split()[0].strip()] = line.split()[1:]
    
    
    vectorFile.close()
    print 'retrieved', len(wordDict.keys())
    return wordDict

# Extracting Vectors from Analogy and Similarity Dataset
validateVectors = vectorExtract()


# In[ ]:

# Dictionary of training pairs for the analogy task
trainDict = dict()
for subDirs in os.listdir(analogyTrainPath):
    for files in os.listdir(analogyTrainPath+subDirs+'/'):
        f = open(analogyTrainPath+subDirs+'/'+files).read().splitlines()
        trainDict[files] = f
print len(trainDict.keys())


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

			validCount = validCount + 1

		else:
			continue

	f1.close()		

	f2 = open(simSummaryFile, "wb")		
	csvwriter = csv.writer(f2, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)

	csvwriter.writerow(['E', len(set(euclideanAnsList) & set(ansWordList)), validCount])
	csvwriter.writerow(['M', len(set(manhattenAnsList) & set(ansWordList)), validCount])
	csvwriter.writerow(['C', len(set(cosineAnsList) & set(ansWordList)), validCount])
		
	f2.close()

	# remember to add MRR

# In[ ]:

def analogyTask(inputDS=analogyDataset,outputFile = anaSoln ): # add more arguments if required

	"""
	Output a file, analogySolution.csv with the following entris
	Query word pair, Correct option, predicted option    
	"""


	return accuracy #return the accuracy of your model after 5 fold cross validation



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
    #similarityTask()
    anaSim = analogyTask()
    #derCos1,derCos2 = derivedWordTask

if __name__ == '__main__':
    main()
