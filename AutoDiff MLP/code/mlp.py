"""
Multilayer Perceptron for character level entity classification
"""
import argparse
import numpy as np
from xman import *
from utils import *
from autograd import *
from functions import *
from copy import deepcopy
import time

np.random.seed(0)

class MLP(object):
    """
    Multilayer Perceptron
    Accepts list of layer sizes [in_size, hid_size1, hid_size2, ..., out_size]
    """
    def __init__(self, layer_sizes):
        self.input_dim = layer_sizes[0]
        self.hidden_dim = layer_sizes[1]
        self.output_dim = layer_sizes[2]
        self.my_xman = self._build() # DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable

    def _build(self):
        x = XMan()

        # Define the model parameters
        seed = np.sqrt(6/(float)(self.input_dim + self.hidden_dim))
        x.w1 = f.param(name='w1', default=np.random.uniform(-seed, seed, (self.input_dim, self.hidden_dim)))
        x.b1 = f.param(name='b1', default=np.random.uniform(-0.1, 0.1, (self.hidden_dim)))
        seed = np.sqrt(6/(float)(self.hidden_dim + self.output_dim))
        x.w2 = f.param(name='w2', default=np.random.uniform(-seed, seed, (self.hidden_dim, self.output_dim)))
        x.b2 = f.param(name='b2', default=np.random.uniform(-0.1, 0.1, (self.output_dim)))

        # Define the input and the output of the network with appropriate dimensions
        x.input_x = f.input(name="input_x", default=np.random.rand(1,self.input_dim))

        # Ensuring that the labels sum to 1 (cross entropy derivative calculation)
        default_y = np.zeros((1, self.output_dim))
        default_y[0, np.random.choice(self.output_dim)] = 1
        x.true_y = f.input(name="true_y", default=default_y)

        # Define the feed forward expressions
        x.o1 = f.relu(f.mul(x.input_x, x.w1) + x.b1)
        x.o2 = f.relu(f.mul(x.o1, x.w2) + x.b2)
        x.probs = f.softMax(x.o2)
        x.loss = f.mean(f.crossEnt(x.probs, x.true_y))

        return x.setup()

def main(params):
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    init_lr = params['init_lr']
    output_file = params['output_file']
    train_loss_file = params['train_loss_file']

    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('%s.train'%dataset, '%s.valid'%dataset, '%s.test'%dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_valid = MinibatchLoader(data.validation, len(data.validation), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)
    mb_test = MinibatchLoader(data.test, len(data.test), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)

    # build
    print "building mlp..."
    mlp = MLP([max_len*mb_train.num_chars,num_hid,mb_train.num_labels])
    #TODO CHECK GRADIENTS HERE


    print "done"

    # train
    print "training..."
    # get default data and params
    value_dict = mlp.my_xman.inputDict()
    lr = init_lr
    train_loss = np.ndarray([0])
    wengert_list = mlp.my_xman.operationSequence(mlp.my_xman.loss)
    min_val_loss = np.inf
    optimal_model = {}
    total_time = 0.0
    
    for i in range(epochs):
        print "Epoch = ", i
        epoch_loss = 0.0
        start_time = time.time()
        #num_batches = 0 
        for (idxs,e,l) in mb_train:
            # Prepare e. Reshape from NxMxV = NxMV
            input_x = e.reshape((e.shape[0], e.shape[1]*e.shape[2]))
            value_dict['input_x'] = input_x
            value_dict['true_y'] = l
            ad = Autograd(mlp.my_xman)

            # Forward Pass
            value_dict = ad.eval(wengert_list, value_dict)

            # Backward Pass
            gradients = ad.bprop(wengert_list, value_dict, loss=np.float_(1.))

            # Update the parameters with the gradients
            for rname in gradients:
                if mlp.my_xman.isParam(rname):
                    value_dict[rname] = value_dict[rname] - (lr * gradients[rname])

            #save the train loss
            epoch_loss += value_dict['loss']
            train_loss = np.append(train_loss, value_dict['loss'])

        end_time = time.time()
        total_time += (end_time - start_time)
        #avg_train_loss = epoch_loss/float(len(mb_train))
        print "Total Training Loss at the end of epoch = ", epoch_loss
                                   
        # validate
        print "Calculating validation loss"
        total_val_loss = 0.0
        for (idxs,e,l) in mb_valid:
            # Prepare and reshape e
            input_x = e.reshape((e.shape[0], e.shape[1]*e.shape[2]))
            value_dict['input_x'] = input_x
            value_dict['true_y'] = l
            ad = Autograd(mlp.my_xman)

             # Forward Pass
            value_dict = ad.eval(wengert_list, value_dict)
            total_val_loss += value_dict['loss']
        #avg_val_loss = total_val_loss/float(len(mb_valid))
        print "Average Validation Loss at the end of epoch = ", total_val_loss

        # Save the optimal model if necessary
        if total_val_loss < min_val_loss:
            min_val_loss = total_val_loss
            optimal_model = deepcopy(value_dict)

    print "Training Done"
    avg_train_time = total_time/float(epochs)
    print "Average time per epoch = ", avg_train_time
    #write out the train loss
    #print "train_loss.shape = ", train_loss.shape
    #print "train_loss.ndim = ", train_loss.ndim
    np.save(train_loss_file, train_loss)    
    
    print "Testing "
    ouput_probabilities = np.ndarray([0])
    total_test_loss = 0.0
    for (idxs,e,l) in mb_test:
        # Prepare and reshape e
        input_x = e.reshape((e.shape[0], e.shape[1]*e.shape[2]))
        #print "e.shape = ", e.shape
        #print "input_x"
        value_dict = optimal_model
        value_dict['input_x'] = input_x
        value_dict['true_y'] = l
        ad = Autograd(mlp.my_xman)

         # Forward Pass
        value_dict = ad.eval(wengert_list, value_dict)
        ouput_probabilities = np.append(ouput_probabilities, value_dict['probs'])
        total_test_loss += value_dict['loss']

    #avg_test_loss = total_test_loss/float(len(mb_test))
    print "Average Test Loss = ", total_test_loss
    # print "mb_test.num_examples = ", mb_test.num_examples
    # print "mb_test.num_labels = ", mb_test.num_labels
    # print "Shape of ouput_probabilities before reshape = ", ouput_probabilities.shape
    ouput_probabilities = np.reshape(ouput_probabilities, (mb_test.num_examples, mb_test.num_labels))
    # print "Shape of ouput_probabilities after reshape = ", ouput_probabilities.shape
        
    # ensure that these are in the same order as the test input
    np.save(output_file, ouput_probabilities)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--dataset', dest='dataset', type=str, default='tiny')
    parser.add_argument('--epochs', dest='epochs', type=int, default=15)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    parser.add_argument('--train_loss_file', dest='train_loss_file', type=str, default='train_loss')
    params = vars(parser.parse_args())
    main(params)
