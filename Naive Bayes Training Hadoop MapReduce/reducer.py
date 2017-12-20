#!/usr/bin/env python
'''
Reducer portion of the MapReduce Framework.
Reads the out output from the sorter that sorts the counter update messages and increments the count of the keys
Stores output in a text file too.
'''
import os
import sys
import re
from ast import literal_eval

# f = open('result.txt', 'wb')

#Prints the messages to STDOUT as a key \t value
def PrintMessage(key, value):
	if key != '':
		out_str = key + '\t' + str(value) + '\n'
		sys.stdout.write(out_str)
		# f.write(out_str)

#Sums the value for the same key and outputs the result
def ProcessCounterUpdates(training_output):
	prev_key = ''
	sum_for_prev_key = 0

	for line in training_output:
		parts = line.split('\t')
		key = parts[0].strip()
		value = parts[1].strip()
		if(key == prev_key):
			sum_for_prev_key += int(value)
		else:
			PrintMessage(prev_key, sum_for_prev_key)
			prev_key = key
			sum_for_prev_key = int(value)

	#Output the last key,value message
	PrintMessage(prev_key, sum_for_prev_key)

ProcessCounterUpdates(sys.stdin)
# f.close()