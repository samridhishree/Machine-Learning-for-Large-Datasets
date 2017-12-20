#!/usr/bin/env python
'''
Mapping portion of the Hadoop MapReduce framework.
Reads the training document one at a time (from sys.stdin) and outputs the counter update messages to the sorting module
'''

import os
import sys
import re
from collections import defaultdict

#Tokenize the documents
def TokenizeDoc(doc):
	return re.findall('\\w+', doc)

#Prints the messages to STDOUT as a key \t value
def PrintMessage(key, value):
	out_str = key + '\t' + str(value) + '\n'
	sys.stdout.write(out_str)

#Parse the input and prepare trianing documents
def OutputUpdateMessages(labels, words):
	out_str = ""
	for label in labels:
		e_y = "Y=" + str(label)
		e_star = "Y=*"
		PrintMessage(e_y ,1)
		PrintMessage(e_star ,1)
		for word in words:
			e_j_y = "Y=" + str(label) + ",W=" + str(word)
			e_star_y = "Y=" + str(label) + ",W=*"
			PrintMessage(e_j_y, 1)
			PrintMessage(e_star_y, 1)

#Read the data from standard input and generate counter update messages
def GenerateCounterUpdates(training_input):
	for line in training_input:
		parts = line.split('\t')
		label = parts[1]
		text = parts[2]
		labels = TokenizeDoc(label)
		words = TokenizeDoc(text)
		# words = [x.lower() for x in words]
		OutputUpdateMessages(labels, words)

GenerateCounterUpdates(sys.stdin.readlines())



