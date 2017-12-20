'''
Classify documents using a Naive Bayes Classifier in Guinea Pig
'''
from guineapig import *
import sys
import math
import re
from collections import defaultdict

# supporting routines can go here

# Storing the stopwords from nltk.corpus.stopwords('english') since autolab does not have nltk.corpus
stopwords = [u'all', u'just', u'being', u'over', u'both', u'through', u'yourselves', u'its', u'before', u'o', u'hadn', u'herself', \
u'll', u'had', u'should', u'to', u'only', u'won', u'under', u'ours', u'has', u'do', u'them', u'his', u'very', u'they', u'not', \
u'during', u'now', u'him', u'nor', u'd', u'did', u'didn', u'this', u'she', u'each', u'further', u'where', u'few', u'because', u'doing', \
u'some', u'hasn', u'are', u'our', u'ourselves', u'out', u'what', u'for', u'while', u're', u'does', u'above', u'between', u'mustn', u't', \
u'be', u'we', u'who', u'were', u'here', u'shouldn', u'hers', u'by', u'on', u'about', u'couldn', u'of', u'against', u's', u'isn', u'or', \
u'own', u'into', u'yourself', u'down', u'mightn', u'wasn', u'your', u'from', u'her', u'their', u'aren', u'there', u'been', u'whom', u'too',\
u'wouldn', u'themselves', u'weren', u'was', u'until', u'more', u'himself', u'that', u'but', u'don', u'with', u'than', u'those', u'he', u'me', \
u'myself', u'ma', u'these', u'up', u'will', u'below', u'ain', u'can', u'theirs', u'my', u'and', u've', u'then', u'is', u'am', u'it', u'doesn', \
u'an', u'as', u'itself', u'at', u'have', u'in', u'any', u'if', u'again', u'no', u'when', u'same', u'how', u'other', u'which', u'you',\
 u'shan', u'needn', u'haven', u'after', u'most', u'such', u'why', u'a', u'off', u'i', u'm', u'yours', u'so', u'y', u'the', u'having', u'once']

'''
Yield lowercase tokens.
'''
def tokens(line): 
    for token in line.split(): 
        yield token.lower()

def tokenize(doc):
	data = re.findall('\\w+', doc)
	# data = [d.lower() for d in data]
	# return data
	for token in data:
		token = token.lower()
		if token not in stopwords:
			yield token

'''
Converting a (label, count) view as a dict= {label:[counts]}
'''
def loadListAsDict(label_list):
	result = defaultdict(list)
	for (key,val) in label_list:
		result[key].append(val)
	return dict(result)

'''
Converting the rows of the view to dict
'''	
def loadViewAsDict(view):
	result = {}
	for key,val in GPig.rowsOf(view):
		result[key] = val
	return result

'''
Yield the labels associated with each document the train set
'''
def getLabelsFromDoc(line):
	labels = line.strip().split('\t')[1].split(',')
	for label in labels:
		yield label

def getLabelAndWords(line):
	parts = line.strip().split('\t')
	labels = parts[1].split(',')
	words = tokenize(parts[2])
	for label in labels:
		yield label, words

'''
Compute the probability and get the predicted label and the proability associated with it
'''
def computePrediction(doc_id, word_label_counts, helper_list):
	[(dom_Y, y_star, dom_X, label_count_dict, label_star_dict)] = helper_list
	prediction = ('label', 0.0)

	# Iterate over all the labels and compute the probabilities associated with each of them
	for label in word_label_counts:
		c_Y_y = label_count_dict[label]
		c_Y_y_W_star = label_star_dict[label]
		prob = 0.0
		for c_Y_y_W_w in word_label_counts[label] :
			prob += math.log(c_Y_y_W_w + 1.0) - math.log(c_Y_y_W_star + dom_X)
		prob += math.log(c_Y_y + 1.0) - math.log(y_star + dom_Y)

		if prob < prediction[1]:
			prediction = (label, prob)

	return(doc_id, prediction[0], prediction[1])

def getLength(words):
	l = 0
	for w in words:
		l += 1
	return l


#always subclass Planner
class NB(Planner):
	# params is a dictionary of params given on the command line. 
	# e.g. trainFile = params['trainFile']
	params = GPig.getArgvParams()
	trainFile = params['trainFile']
	testFile = params['testFile']

	'''
	The number of times each label occurs in the training set
	The view labels_in_doc is of the form: [('Agent', 135), ..... ,('Person', 849)] ==> Equivalent for C[Y=y']
	'''
	train_data = ReadLines(trainFile)
	labels_in_doc = Flatten(train_data, by=getLabelsFromDoc) | Group(by=lambda x:x, reducingTo= ReduceToCount())

	# Total number of labels in the dataset ==> equivalent for |y| or dom(Y)
	dom_Y = Group(labels_in_doc, by=lambda x:'dom_y', reducingTo= ReduceToCount())

	# Equivalent of c[Y=*]
	y_star = Group(labels_in_doc, by=lambda x:'y=*', reducingTo= ReduceTo(int, by=lambda accum, (label,count): accum+count))

	'''
	Generate word and label counts associated with it.
	The output view should be of form: (word1, [(label1, count1), (label2, count2),....])
	''' 
	label_words = Flatten(train_data, by=getLabelAndWords) \
				  | Flatten(by= lambda (label, words): map(lambda w : (label, w), words)) \
				  | Group(by=lambda x:x, reducingTo=ReduceToCount()) \
				  | Group(by= lambda ((label, word), count): word, retaining= lambda ((label, word), count):(label, count), reducingTo = ReduceToList())

	'''
	Generate counts equivalent to C[Y=y',W=*] for all labels across the training set
	Output view if of the form (label, count)
	'''
	label_star = Flatten(train_data, by=getLabelAndWords) \
				| Map(by=lambda (label, words):(label, getLength(words))) \
				| Group(by=lambda (label,count):label, reducingTo=ReduceTo(int, by=lambda accum, (label,count): accum+count))
				
	# Get the train side vocabulary ==> Equivalent for |V|
	dom_X = Group(label_words, by=lambda (word,label_list):'dom_x', reducingTo= ReduceToCount())

	# Read the test document. The label is not included and the view is [(docid, word), (docid, word) ...]
	test_doc = ReadLines(testFile) \
			 | Map(by=lambda line: line.strip().split("\t")[0::2]) \
			 | Map(by= lambda (doc_id, doc): (doc_id, tokenize(doc))) \
			 | Flatten(by= lambda (doc_id, words): map(lambda w : (doc_id, w), words))

	'''
	Join the test_doc view with the label_words view to get a combined representation.
	Join is performed on the words.
	The output view should look like: (docid, {label1:[count11, count12...], label2:[count21, count22,...], ...}, ....)
	'''
	doc_label_count_dict = Join(Jin(test_doc, by=lambda (docid, word): word), Jin(label_words, by=lambda (word, label_counts): word)) \
	 				  | Group(by=lambda ((docid, word1), (word2, label_counts)): docid, \
					  		retaining= lambda ((docid, word1), (word2, label_counts)) : label_counts, \
					  		reducingTo= ReduceTo(list, by=lambda accum, val : accum + val)) \
					  | Map(by= lambda (docId, label_counts) : (docId, loadListAsDict(label_counts)))

	'''
	Create a view of the form: (dom_Y, y_star, dom_X, {Individual label counts (C[Y=y'])}, {Individual label word counts (C[Y=y', W=*])}).
	This view is essential to copy information in order to calculate the probabilities
	To save space removing labels like "dom_Y", "dom_X" etc
	'''
	helper_join = Join(Jin(dom_Y, by= lambda row: 'helper'), Jin(y_star, by= lambda row: 'helper'), Jin(dom_X, by= lambda row: 'helper')) \
				  | Augment(sideview= labels_in_doc, loadedBy = loadViewAsDict ) \
				  | Augment(sideview= label_star, loadedBy = loadViewAsDict ) \
				  | Map(by= lambda (((dom_Y, y_star, dom_X), label_count_dict), label_star_dict) : \
				  	(float(dom_Y[-1]), float(y_star[-1]), float(dom_X[-1]), label_count_dict, label_star_dict))

	output = Augment(doc_label_count_dict, sideview= helper_join ) \
			 | Map(by= lambda ((doc_id, word_label_counts), helper_list) : computePrediction(doc_id, word_label_counts, helper_list))


# always end like this
if __name__ == "__main__":
    NB().main(sys.argv)

# supporting routines can go here
