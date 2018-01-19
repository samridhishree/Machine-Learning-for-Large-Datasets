# Document Classifier in GuineaPIG

This project contains an implementation of a document classifier (Naive Bayes) using the Hadoop Framework GuineaPIG.

## Data
The dataset is extracted from DBpedia. The labels of the article are based on the types of the document. There are in total 17 (16 + other) classes in the dataset, and they are from the first level class in DBpedia ontology. The training data format is one document per line. Each line contains three columns which are separated by a single tab:

* a document id
* a comma separated list of class labels
* document words

The testing data format is one document per line. Each line contains two columns which are separated by a single tab:
* a document id
* document words
The documents are preprocessed so that there are no tabs in the body.

## GuinePIG
GuineaPig is a lightweight Python library that is similar to Pig. The workflows in GuineaPIG are expressed  using high level constructs (such as Join, Augment etc.) and it spawns off MapReduce tasks behind the scenes to do the compute. GuineaPig provides for you a layer of abstraction over bare-bones Hadoop. More information about GuineaPig (including a tutorial) can be found at http://curtis.ml.cmu.edu/w/courses/index.php/Guinea_Pig.