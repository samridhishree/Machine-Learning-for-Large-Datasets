"""
Long Short Term Memory for character level entity classification
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

class LSTM(object):
    """
    Long Short Term Memory + Feedforward layer
    Accepts maximum length of sequence, input size, number of hidden units and output size
    """
    def __init__(self, max_len, in_size, num_hid, out_size):
        self.input_dim = in_size
        self.hidden_dim = num_hid
        self.output_dim = out_size
        self.max_len = max_len
        self.my_xman = self._build() #DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable

    def _build(self):
        x = XMan()
        # Define the model parameters
        seed = np.sqrt(6/(float)(self.input_dim + self.hidden_dim))

        # Input Gate
        x.wi = f.param(name='wi', default=np.random.uniform(-seed, seed, (self.input_dim, self.hidden_dim)))
        x.ui = f.param(name='ui', default=np.random.uniform(-seed, seed, (self.hidden_dim, self.hidden_dim)))
        x.bi = f.param(name='bi', default=np.random.uniform(-0.1, 0.1, (self.hidden_dim)))

        # Forget Gate
        x.wf = f.param(name='wf', default=np.random.uniform(-seed, seed, (self.input_dim, self.hidden_dim)))
        x.uf = f.param(name='uf', default=np.random.uniform(-seed, seed, (self.hidden_dim, self.hidden_dim)))
        x.bf = f.param(name='bf', default=np.random.uniform(-0.1, 0.1, (self.hidden_dim)))

        # Output Gate
        x.wo = f.param(name='wo', default=np.random.uniform(-seed, seed, (self.input_dim, self.hidden_dim)))
        x.uo = f.param(name='uo', default=np.random.uniform(-seed, seed, (self.hidden_dim, self.hidden_dim)))
        x.bo = f.param(name='bo', default=np.random.uniform(-0.1, 0.1, (self.hidden_dim)))

        # Context
        x.wc = f.param(name='wc', default=np.random.uniform(-seed, seed, (self.input_dim, self.hidden_dim)))
        x.uc = f.param(name='uc', default=np.random.uniform(-seed, seed, (self.hidden_dim, self.hidden_dim)))
        x.bc = f.param(name='bc', default=np.random.uniform(-0.1, 0.1, (self.hidden_dim)))

        seed = np.sqrt(6/(float)(self.hidden_dim + self.output_dim))
        x.w2 = f.param(name='w2', default=np.random.uniform(-seed, seed, (self.hidden_dim, self.output_dim)))
        x.b2 = f.param(name='b2', default=np.random.uniform(-0.1, 0.1, (self.output_dim)))

        # Create a sequence of inputs
        for i in range(self.max_len):
            x_attr = 'x' + str(i)
            #x.x_attr = f.input(name=x_attr, default=np.random.rand(1,self.input_dim))
            setattr(x, x_attr, f.input(name=x_attr, default=np.random.rand(1,self.input_dim)))
        #x.batch_size = f.input(name="batch_size", default=1)

        x.h0 = f.input(name='h0', default=np.zeros((self.hidden_dim)))
        x.c0 = f.input(name='c0', default=np.zeros((self.hidden_dim)))

        # Make the LSTM sequential updates
        for t in xrange(1, self.max_len+1):
            x_attr = 'x' + str(t-1)
            h_attr = 'h' + str(t-1)
            prev_c_attr = 'c' + str(t-1)
            f_attr = 'f' + str(t)
            c_cap = 'c_cap' + str(t)
            i_attr = 'i' + str(t)
            o_attr = 'o' + str(t)
            cur_c_attr = 'c' + str(t)
            cur_x = getattr(x, x_attr)
            prev_h = getattr(x, h_attr)
            prev_c = getattr(x, prev_c_attr)
            setattr(x, 'i'+str(t), f.sigmoid(f.add(f.add(f.mul(cur_x, x.wi), f.mul(prev_h, x.ui)), x.bi)))
            setattr(x, 'f'+str(t), f.sigmoid(f.add(f.add(f.mul(cur_x, x.wf), f.mul(prev_h, x.uf)), x.bf)))
            setattr(x, 'o'+str(t), f.sigmoid(f.add(f.add(f.mul(cur_x, x.wo), f.mul(prev_h, x.uo)), x.bo)))
            setattr(x, 'c_cap'+str(t), f.tanh(f.add(f.add(f.mul(cur_x, x.wc), f.mul(prev_h, x.uc)), x.bc)))
            setattr(x, 'c'+str(t), f.add(f.hadamard(getattr(x, f_attr), prev_c), f.hadamard(getattr(x, i_attr), getattr(x, c_cap))))
            setattr(x, 'h'+str(t), f.hadamard(getattr(x, o_attr), f.tanh(getattr(x, cur_c_attr))))

        # Ensuring that the labels sum to 1 (cross entropy derivative calculation)
        default_y = np.zeros((1, self.output_dim))
        default_y[0, np.random.choice(self.output_dim)] = 1
        x.true_y = f.input(name="true_y", default=default_y)

        # Define the feed forward expressions
        #x.h_t = self._lstm(x)
        h_m = 'h' + str(t)
        x.o2 = f.relu(f.mul(getattr(x, h_m), x.w2) + x.b2)
        x.probs = f.softMax(x.o2)
        x.loss = f.mean(f.crossEnt(x.probs, x.true_y))

        return x.setup()

def update_sequence(matrix_x):
    M = matrix_x.shape[1]
    output_x = []
    for i in range(M):
        cur_x = matrix_x[:, i, :]
        cur_x = np.flip(cur_x, axis=1)
        output_x.append(cur_x)
    return output_x


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
    print "building lstm..."
    lstm = LSTM(max_len,mb_train.num_chars,num_hid,mb_train.num_labels)
    #OPTIONAL: CHECK GRADIENTS HERE


    print "done"

    # train
    print "training..."
    # get default data and params
    value_dict = lstm.my_xman.inputDict()
    lr = init_lr
    train_loss = np.ndarray([0])
    wengert_list = lstm.my_xman.operationSequence(lstm.my_xman.loss)
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
            #print "shape of e = ", e.shape
            cur_batch_size = e.shape[0]
            #print "current batch size = ", cur_batch_size
            # Set the input sequence values
            for i in xrange(1, max_len+1):
                cur_x = e[:, i-1, :]
                x_attr = 'x' + str(max_len-i)
                value_dict[x_attr] = cur_x
            value_dict['true_y'] = l
            ad = Autograd(lstm.my_xman)

            # Forward Pass
            value_dict = ad.eval(wengert_list, value_dict)

            # Backward Pass
            gradients = ad.bprop(wengert_list, value_dict, loss=np.float_(1.))

            # Update the parameters with the gradients
            for rname in gradients:
                if lstm.my_xman.isParam(rname):
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
            #input_x = e.reshape((e.shape[0], e.shape[1]*e.shape[2]))
            for i in xrange(1, max_len+1):
                cur_x = e[:, i-1, :]
                x_attr = 'x' + str(max_len-i)
                value_dict[x_attr] = cur_x
            value_dict['true_y'] = l
            ad = Autograd(lstm.my_xman)

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
        # Prepare
        value_dict = optimal_model
        for i in xrange(1, max_len+1):
            cur_x = e[:, i-1, :]
            x_attr = 'x' + str(max_len-i)
            value_dict[x_attr] = cur_x
        value_dict['true_y'] = l
        ad = Autograd(lstm.my_xman)

         # Forward Pass
        value_dict = ad.eval(wengert_list, value_dict)
        ouput_probabilities = np.append(ouput_probabilities, value_dict['probs'])
        total_test_loss += value_dict['loss']

    print "Average Test Loss = ", total_test_loss
    ouput_probabilities = np.reshape(ouput_probabilities, (mb_test.num_examples, mb_test.num_labels))
        
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
