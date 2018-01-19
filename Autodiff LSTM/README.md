# Automatic Reverse-Mode Differentiation (Autodiff) for LSTM

This is am implementation of Automatic Reverse-Mode Differentiation for LSTM for character-level entity classification.  Character level entity classification refers to determining the type of an entity given the characters which appear in its name as features. For example, given the name “Antonio Veciana" you might guess that it is a Person, and given the name “Anomis esocampta" you might guess that it is a Species. This implementation is trained to classify following 5 DBPedia categories - Person, Place, Organisation, Work, Species.

## Data
The data contains two columns separated by a tab, with the title in the first column and label in second. The characters in the title are converted to a one-hot representation. The maximum length of an entity is a tenable hyper-parameter (longer entities are truncated to this length and shorter ones are padded with white space). 

__*For more details on Autodiff and Wengert Lists please visit:*__ http://www.bcl.hamilton.ie/~barak/papers/toplas-reverse.pdf or https://justindomke.wordpress.com/2009/03/24/a-simple-explanation-of-reverse-mode-automatic-differentiation/

__*For general backprop:*__ http://colah.github.io/posts/2015-08-Backprop/

__*For more on LSTMs:*__ http://colah.github.io/posts/2015-08-Understanding-LSTMs/