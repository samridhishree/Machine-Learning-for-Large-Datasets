'''
Logistic Regression Classifier for document classification using lasy regularized SGD updates
'''

import os
import sys
import math
import re

# Read all the command line arguments
vocab_size = int(sys.argv[1])
learning_rate = float(sys.argv[2])
decay = float(sys.argv[3])
epochs = int(sys.argv[4])
train_size = int(sys.argv[5])
test_file = sys.argv[6]

# Define and the hash tables and variables required
num_labels = 5
all_labels = []
B = [[0.0]*vocab_size for i in range(num_labels)]	#weights hashmap, initialize to 0
A = [[0]*vocab_size for i in range(num_labels)]		#last update hashmap, initialize to 0


# Function to calculate the sigmoid of a value. Prevents under/over flow
def sigmoid(score):
	overflow = 20.0
	if score > overflow:
		score = overflow
	elif score < -overflow:
		score = -overflow

	exp = math.exp(score)
	return exp/(1.0+exp)

# Return the tokens present in the line
def getTokens(line):
	tokens = re.findall('\\w+', line)
	return tokens

'''
Tokenize the documents and return the hash values of the words present.
The return values contains the integer hash code for each word in the document
'''
def tokenizeDoc(doc):
	words = getTokens(doc)
	words = [x.lower() for x in words]
	num_words = len(words)
	features = [0] * num_words
	for i in range(num_words):
		word_hash = int(hash(words[i]) % vocab_size)
		features[i] = (word_hash + vocab_size) if word_hash < 0 else word_hash
	return features

# Processes each document read from command line or file ans returns the labels and features
def processDocument(doc):
	parts = doc.split('\t')
	labels = parts[1]
	text = parts[2]
	labels = getTokens(labels)
	features = tokenizeDoc(text)
	return labels, features

'''
Function to train the model. 
Reads the imput from the input stream.
The number of epochs are determined by the train_size passed as a param.
Does a lazy SGD update
'''
def train(training_input):
	global num_labels
	global learning_rate
	global decay
	global epochs
	global B
	global A
	k = 0
	epoch = 1
	lamda = learning_rate
	mu = decay

	# Begin training
	for line in training_input:
		k += 1
		labels, features = processDocument(line)
		# Add the labels if not already added
		if len(all_labels) != num_labels:
			for label in labels:
				if label not in all_labels:
					all_labels.append(label)

		# Convert labels to integer indexes
		labels = [all_labels.index(label) for label in labels]
		#print "labels = ", all_labels

		# Perform weight updates for all the labels
		for i in range(num_labels):
			# Compute the probability
			dot_product = 0.0
			for j in features:
				dot_product += B[i][j]
			p = sigmoid(dot_product)
			y = 1.0 if i in labels else 0.0
			if abs(y-p) < 1e-4:
				continue
			
			# Update the weights for non-zero features
			for j in features:
				if abs(B[i][j]) >= 0.8:
					B[i][j] *=  pow((1.0 - (lamda * mu)), float(k - A[i][j]))
				B[i][j] +=  lamda * (y - p)
				A[i][j] = k
			
		# Check if one epoch is over
		if (k % train_size) == 0:
			#Check if it reached final epoch
			if epoch == epochs:
				#Update all the params again for the last update
				for i in range(num_labels):
					for j in range(vocab_size):
						if abs(B[i][j]) >= 0.8:
							B[i][j] *=  pow((1.0 - (lamda * mu)), float(k - A[i][j]))
						#B[i][j] *= pow((1.0 - (lamda * mu)), float(k - A[i][j]))
			else:
				#avg_prob = [prob/float(train_size) for prob in total_prob]
				#sys.stdout.write("Average Probability per class at end of epoch " + str(epoch) + " = " + str(avg_prob) + '\n')
				epoch += 1
				lamda = learning_rate/float((epoch * epoch))
				#sys.stdout.write("Learning rate updated = " + str(lamda) + '\n')
				#sys.stdout.write("Epoch : " + str(epoch) + '\n')
				#total_prob = [0.0] * num_labels


'''
Function to test the model.
Predicts the document labels for the test data
'''
def test(test_file):
	global num_labels
	global all_labels
	global B
	f = open(test_file, 'rb')
	#correct = 0
	#num_docs = 0
	for line in f:
		#num_docs += 1
		labels, features = processDocument(line)
		#print "before index = ", labels
		labels = [all_labels.index(label) for label in labels]
		#print "after index = ", labels
		#print "features = ", features
		#predicted = []

		# Calculate the probability for each label
		for i in range(num_labels):
			# Compute the probability
			dot_product = 0.0
			p = 0.0
			for j in features:
				dot_product += B[i][j]
			p = sigmoid(dot_product)
			#predicted.append(p)

			# Print the output in the required format
			sys.stdout.write(all_labels[i] + '\t' + str(p))
			if i < num_labels-1:
				sys.stdout.write(",")
		sys.stdout.write("\n")

		# Calculate the accuracy
		#pred = predicted.index(max(predicted))
		#print "pred = ", predicted
		#print "truth = ", labels
		#if pred in labels:
			#correct += 1
	#acc = correct/float(num_docs)
	#sys.stdout.write("Accuracy = " + str(acc) + '\n')


training_input = sys.stdin
print "Starting the training: "
train(training_input)
print "Training Finished. Testing the model"
test(test_file)















