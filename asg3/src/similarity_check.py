import numpy as np
import os, sys
from itertools import chain

def get_vocab_dict(filename, que_words, choices):
	if not os.path.exists(os.path.join(os.getcwd(), "que_wordvec.npy")):
		que_wordvec = {}
		choices_wordvec = {}
		f = open(filename, 'rb')
		
		for line in f:
			word = line.split(' ')[0]
			#vocab_file = open("vocab_list.txt", "a")
			#vocab_file.write(word + "\n")
			if word in que_words:
				que_wordvec.update({word: np.array(line.strip().split(' ')[1:]).astype(np.float32)})
			if word in choices:
				choices_wordvec.update({word: np.array(line.strip().split(' ')[1:]).astype(np.float32)})
			else:
				pass
			#vocab_file.close()
		f.close()	
		np.save("que_wordvec.npy", que_wordvec)
		np.save("choices_wordvec.npy", choices_wordvec)
		return que_wordvec, choices_wordvec	
	
	else:
		que_wordvec = np.load("que_wordvec.npy").item()
		choices_wordvec = np.load("choices_wordvec.npy").item()
		return que_wordvec, choices_wordvec
	


def get_que_ans(filename):
	que_words = []
	ans_words = []
	choices = []
	f = open(filename, 'rb')
	for line in f:
		que_words.append(line.split(' | ')[0])
		ans_words.append(line.split(' | ')[1])
		choices.append(line.strip().split(' | ')[1:])
	f.close()
	return que_words, ans_words, choices

def euclidean_distance(wordvec1, wordvec2):
	return np.linalg.norm(wordvec1 - wordvec2) 


def give_similar_word(word, choice_list, que_wordvec, choices_wordvec):
	dist_list=[]
	for i in range(len(choice_list)):
		#min_dist = 100.0	
		if (choice_list[i] in choices_wordvec.keys()) and (word in que_wordvec.keys()):
			dist_list.append(euclidean_distance(que_wordvec[word], choices_wordvec[choice_list[i]]))
		else:
			print "Question invalid for | " + word
			return 0

	return choice_list[np.argmin(dist_list)]	
			  	


# def get_word_vec(word):
# 	f1 = open("vocab_list.txt", 'rb')
# 	f2 = open('../data_q1/glove.6B.300d.txt', 'rb')
# 	for index, line in enumerate(f1):
# 		if word == line.strip():
# 			for v_index, v_line in enumerate(f2):
# 				if v_index == index:
# 					word_vec = v_line.strip().split(' ')[1:]
# 				else:
# 					pass	

# 		else:
# 			pass	

	
que_words, ans_words, choices = get_que_ans('Q1/word-similarity-dataset')
choices_flat = [x for sublist in choices for x in sublist]

que_wordvec, choices_wordvec = get_vocab_dict("Q1/glove.6B.300d.txt", que_words, choices_flat)
#print choices_wordvec.values()[5]
output_words = []
for index, word in enumerate(que_words):
	similar_word = give_similar_word(word, choices[index], que_wordvec, choices_wordvec)
	if similar_word:
		print "similar word to " + word + " is " + similar_word
		output_words.append(similar_word)
	else:
		output_words.append(0)
		pass	

print output_words

# compare output_words and ans_words
