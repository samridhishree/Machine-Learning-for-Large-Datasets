# Naive Bayes Classifier in MapReduce Framework

This project contains an implementation of Naive Bayes training in the Hadoop Map Reduce Framework using python for document classification. The training is split into a mapper and a reducer. The output of the reducer is the aggregated counts of the words that appear in a training corpus

## Data
The dataset is extracted from DBpedia. The labels of the article are based on the types of the document. There are in total 17 (16 + other) classes in the dataset, and they are from the first level class in DBpedia ontology. The training data format is one document per line. Each line contains three columns which are separated by a single tab:

* a document id
* a comma separated list of class labels
* document words

The testing data format is one document per line. Each line contains two columns which are separated by a single tab:
* a document id
* document words
The documents are preprocessed so that there are no tabs in the body. We consider only 5 classes for training.
