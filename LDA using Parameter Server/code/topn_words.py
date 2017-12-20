import os
import sys
import pandas as pd
import heapq
import csv
import operator


word_topic_file = sys.argv[1]
vocab_file = sys.argv[2]
output_file = sys.argv[3]
N = 10

#Build vocabulary
vocab_data = pd.read_csv(vocab_file, header=None)
vocab = list(vocab_data[0])
data = pd.read_csv(word_topic_file, header=None)
f = open(output_file, 'wb')
writer = csv.writer(f)

for topic in range(data.columns.shape[0]):
    to_write = [topic]
    all_words = list(data[topic])
    top_words = zip(*heapq.nlargest(N, enumerate(all_words), key=operator.itemgetter(1)))[0]
    top_words = [vocab[word] for word in top_words]
    to_write.extend(top_words)
    writer.writerow(to_write)
f.close()
