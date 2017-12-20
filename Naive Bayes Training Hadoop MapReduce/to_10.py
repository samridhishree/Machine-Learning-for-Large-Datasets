'''
Get the top 10 words for every class tag
'''

import os
import sys
from collections import defaultdict

def ParseInput(input_str):
	part = input_str.split('\t')
	key = part[0].strip()
	value = int(part[1].strip())
	label = ''
	word = ''
	if("W=" in key and "W=*" not in key):
		sub_part = key.split(',')
		label = (sub_part[0].strip('Y='))
		word = (sub_part[1].strip('W='))
	return label, word, value

def dd():
	return defaultdict(int)

input_file = sys.argv[1]
top_10 = defaultdict(dd)
f = open(input_file, 'rb')

#Create a dictionary of top 10 words for each class - something similar to MinHeap method
for line in f:
	label, word, value = ParseInput(line)
	if label == '':
		continue	
	label_dict = top_10[label]
	if(len(label_dict.keys()) == 10):
		min_key = min(label_dict, key=label_dict.get)
		min_value = label_dict[min_key]
		if value > min_value:
			top_10[label].pop(min_key, None)
			top_10[label][word] = value
	else:
		top_10[label][word] = value
f.close()

writer = open('top10_tiny.txt', 'w')
for label in top_10:
	# if(len(top_10[label].keys()) == 10):
	for word in top_10[label]:
		writer.write(label + '\t' + word + '\t' + str(top_10[label][word]) + '\n')
writer.close()




