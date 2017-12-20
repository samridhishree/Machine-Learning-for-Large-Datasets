import io
import numpy as np
import urllib

def clean(x):
    return urllib.unquote(str(x)).decode('utf8').strip()

class Data:
    def __init__(self, training, validation, test, chardict, labeldict):
        self.chardict = chardict
        self.labeldict = labeldict
        self.training = training
        self.test = test
        self.validation = validation

class DataPreprocessor:
    def preprocess(self, train_file, validation_file, test_file):
        """
        preprocess train and test files into one Data object.
        construct character dict from both
        """
        chardict, labeldict = self.make_dictionary(train_file, validation_file, test_file)
        print 'preparing training data'
        training = self.parse_file(train_file, chardict, labeldict)
        
        print 'preparing validation data'
        validation = self.parse_file(validation_file, chardict, labeldict)

        print 'preparing test data'
        test = self.parse_file(test_file, chardict, labeldict)

        return Data(training, validation, test, chardict, labeldict)

    def make_dictionary(self, train_file, validation_file, test_file):
        """
        go through train and test data and get character and label vocabulary
        """
        print 'constructing vocabulary'
        train_set, test_set, valid_set = set(), set(), set()
        label_set = set()
        ftrain = io.open(train_file, 'r')
        for line in ftrain:
            entity, label = map(clean, line.rstrip().split('\t')[:2])
            train_set |= set(list(entity))
            label_set |= set(label.split(','))

        fvalid = io.open(validation_file, 'r')
        for line in fvalid:
            entity, label = map(clean, line.rstrip().split('\t')[:2])
            valid_set |= set(list(entity))
            label_set |= set(label.split(','))

        ftest = io.open(test_file, 'r')
        for line in ftest:
            entity, label = map(clean, line.rstrip().split('\t')[:2])
            test_set |= set(list(entity))
        
        print '# chars in training ', len(train_set)
        print '# chars in validation ', len(valid_set)
        print '# chars in testing ', len(test_set)
        print '# chars in (testing-training-validation) ', len(test_set-train_set-valid_set)
        print '# labels', len(label_set)

        vocabulary = list(train_set | test_set | valid_set)
        vocab_size = len(vocabulary)
        chardict = dict(zip(vocabulary, range(1,vocab_size+1)))
        chardict[u' '] = 0
        labeldict = dict(zip(list(label_set), range(len(label_set))))
        
        return chardict, labeldict

    def parse_file(self, infile, chardict, labeldict):
        """
        get all examples from a file. 
        replace characters and labels with their lookup
        """
        examples = []
        fin = io.open(infile, 'r')
        # idx is for the index of the row in the 
        # original file before shuffling and randomization
        idx = 0
        for line in fin:                
            entity, label = map(clean, line.rstrip().split('\t')[:2])
            # print entity
            ent = map(lambda c:chardict[c], list(entity))
            lab = map(lambda l:labeldict[l] if l in labeldict else 0, label.split(','))
            examples.append((idx, ent, lab))
            idx += 1
        fin.close()
        print "num_rows:", len(examples), " index", idx
        return examples

class MinibatchLoader:
    def __init__(self, examples, batch_size, max_len, num_chars, num_labels, shuffle=True):
        self.batch_size = batch_size
        self.max_len = max_len
        self.examples = examples
        self.num_examples = len(examples)
        self.num_labels = num_labels
        self.num_chars = num_chars
        self.shuffle = shuffle
        self.reset()

    def __iter__(self):
        """ make iterable """
        return self

    def reset(self):
        """ next epoch """
        if self.shuffle: self.permutation = np.random.permutation(self.num_examples)
        else: self.permutation = np.arange(self.num_examples)
        self.ptr = 0

    def next(self):
        """ get next batch of examples """
        if self.ptr >= self.num_examples:
            self.reset()
            raise StopIteration()
        batch_size = self.batch_size
        if self.ptr>self.num_examples-self.batch_size:
            batch_size = self.num_examples-self.ptr

        ixs = range(self.ptr,self.ptr+batch_size)
        self.ptr += batch_size

        i = np.zeros((batch_size), dtype='int32')
        e = np.zeros((batch_size, self.max_len, 
            self.num_chars), dtype='int32') # entity
        l = np.zeros((batch_size, self.num_labels), dtype='int32') # labels
        for n, ix in enumerate(ixs):
            idx, ent, lab = self.examples[self.permutation[ix]]
            le = min(len(ent),self.max_len)
            i[n] = idx
            e[n,np.arange(le),ent[:le]] = 1
            l[n,lab] = 1

        return i, e, l


