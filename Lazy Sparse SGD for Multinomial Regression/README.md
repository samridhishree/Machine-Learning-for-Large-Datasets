# Multinomial Regression with Lazy Sparse Stochastic Gradient Descent

This project contains an implementation of Lazy Sparse Stochastic Gradient Descent for Regularized Mutlinomial Logistic Regression for document classification. The multilevel classification problem here is treated as multiple independent binary classification tasks. 5 document classes are used for training and testing. 

## Data
The dataset is extracted from DBpedia. The labels of the article are based on the types of the document. There are in total 17 (16 + other) classes in the dataset, and they are from the first level class in DBpedia ontology. The training data format is one document per line. Each line contains three columns which are separated by a single tab:

* a document id
* a comma separated list of class labels
* document words

The testing data format is one document per line. Each line contains two columns which are separated by a single tab:
* a document id
* document words
The documents are preprocessed so that there are no tabs in the body. We consider only 5 classes for training.


__*For more details on lazy SGD please visit:*__ https://lingpipe.files.wordpress.com/2008/04/lazysgdregression.pdf