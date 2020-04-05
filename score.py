from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from torch.autograd import Variable
import os
import warnings
import math
import numpy as np
from tqdm import tqdm, trange
import time
import random
import csv
from sklearn.ensemble import RandomForestRegressor as RFR
import rdkit
# import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import QED
from joblib import dump, load

import threading

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

from sklearn.externals import joblib
import pickle


def normalize_desc(desc_array, desc_mean=None):
    desc_array = np.array(desc_array).reshape(len(desc_array), -1)
    ind = np.zeros(desc_array.shape)
    for i in range(desc_array.shape[0]):
        for j in range(desc_array.shape[1]):
            try:
                if np.isfinite(desc_array[i, j]):
                    ind[i, j] = 1
            except:
                pass
    for i in range(desc_array.shape[0]):
        for j in range(desc_array.shape[1]):
            if ind[i, j] == 0:
                desc_array[i, j] = 0
    if desc_mean is None:
        desc_mean = np.mean(desc_array, axis=0)
    for i in range(desc_array.shape[0]):
        for j in range(desc_array.shape[1]):
            if ind[i, j] == 0:
                desc_array[i, j] = desc_mean[j]
    return desc_array, desc_mean
class Iterator(object):
    """Abstract base class for data iterators.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)
        if n < batch_size:
            raise ValueError('Input data length is shorter than batch_size\nAdjust batch_size')

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)
class SmilesIterator(Iterator):
    """Iterator yielding data from a SMILES array.
    # Arguments
        x: Numpy array of SMILES input data.
        y: Numpy array of targets data.
        smiles_data_generator: Instance of `SmilesEnumerator`
            to use for random SMILES generation.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        dtype: dtype to use for returned batch. Set to keras.backend.floatx if using Keras
    """

    def __init__(self, x, y, smiles_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dtype=np.float32
                 ):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        self.x = np.asarray(x)

        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.smiles_data_generator = smiles_data_generator
        self.dtype = dtype
        super(SmilesIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(
            tuple([current_batch_size] + [self.smiles_data_generator.pad, self.smiles_data_generator._charlen]),
            dtype=self.dtype)
        for i, j in enumerate(index_array):
            smiles = self.x[j:j + 1]
            x = self.smiles_data_generator.transform(smiles)
            batch_x[i] = x

        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y
def get_desc(smiles, calc):
    desc = []
    processed_indices = []
    invalid_indices = []
    for i in range(len(smiles)):
        sm = smiles[i]
        try:
            mol = Chem.MolFromSmiles(sm)
            tmp = np.array(calc(mol))
            desc.append(tmp)
            processed_indices.append(i)
        except:
            invalid_indices.append(i)

    desc_array = np.array(desc)
    return desc_array, processed_indices, invalid_indices

def sanitize_smiles(smiles, canonical=True, throw_warning=False):
    """
    Takes list of SMILES strings and returns list of their sanitized versions.
    For definition of sanitized SMILES check
    http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol
    Parameters
    ----------
    smiles: list
        list of SMILES strings
    canonical: bool (default True)
        parameter specifying whether SMILES will be converted to canonical
        format
    throw_warning: bool (default False)
        parameter specifying whether warnings will be thrown if a SMILES is
        invalid
    Returns
    -------
    new_smiles: list
        list of SMILES and NaNs if SMILES string is invalid or unsanitized.
        If canonical is True, returns list of canonical SMILES.
    When canonical is True this function is analogous to:
        canonical_smiles(smiles, sanitize=True).
    """
    new_smiles = []
    for sm in smiles:
        try:
            if canonical:
                new_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(sm, sanitize=True)))
            else:
                new_smiles.append(sm)
        except:
            if throw_warning:
                warnings.warn('Unsanitized SMILES string: ' + sm, UserWarning)
            new_smiles.append('')
    return new_smiles


def canonical_smiles(smiles, sanitize=True, throw_warning=False):
    """
    Takes list of SMILES strings and returns list of their canonical SMILES.
    Parameters
    ----------
    smiles: list
        list of SMILES strings to convert into canonical format
    sanitize: bool (default True)
        parameter specifying whether to sanitize SMILES or not.
            For definition of sanitized SMILES check
            http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol
    throw_warning: bool (default False)
        parameter specifying whether warnings will be thrown if a SMILES is
        invalid
    Returns
    -------
    new_smiles: list
        list of canonical SMILES and NaNs if SMILES string is invalid or
        unsanitized (when sanitize is True)
    When sanitize is True the function is analogous to:
        sanitize_smiles(smiles, canonical=True).
    """
    new_smiles = []
    for sm in smiles:
        try:
            mol = Chem.MolFromSmiles(sm, sanitize=sanitize)
            new_smiles.append(Chem.MolToSmiles(mol))
        except:
            if throw_warning:
                warnings.warn(sm + ' can not be canonized: invalid '
                                   'SMILES string!', UserWarning)
            new_smiles.append('')
    return new_smiles


def save_smi_to_file(filename, smiles, unique=True):
    """
    Takes path to file and list of SMILES strings and writes SMILES to the specified file.
        Args:
            filename (str): path to the file
            smiles (list): list of SMILES strings
            unique (bool): parameter specifying whether to write only unique copies or not.
        Output:
            success (bool): defines whether operation was successfully completed or not.
       """
    if unique:
        smiles = list(set(smiles))
    else:
        smiles = list(smiles)
    f = open(filename, 'w')
    for mol in smiles:
        f.writelines([mol, '\n'])
    f.close()
    return f.closed


def read_smi_file(filename, unique=True, add_start_end_tokens=False):
    """
    Reads SMILES from file. File must contain one SMILES string per line
    with \n token in the end of the line.
    Args:
        filename (str): path to the file
        unique (bool): return only unique SMILES
    Returns:
        smiles (list): list of SMILES strings from specified file.
        success (bool): defines whether operation was successfully completed or not.
    If 'unique=True' this list contains only unique copies.
    """
    f = open(filename, 'r')
    molecules = []
    for line in f:
        if add_start_end_tokens:
            molecules.append('<' + line[:-1] + '>')
        else:
            molecules.append(line[:-1])
    if unique:
        molecules = list(set(molecules))
    else:
        molecules = list(molecules)
    f.close()
    return molecules, f.closed


def tokenize(smiles, tokens=None):
    """
    Returns list of unique tokens, token-2-index dictionary and number of
    unique tokens from the list of SMILES
    Parameters
    ----------
        smiles: list
            list of SMILES strings to tokenize.
        tokens: list, str (default None)
            list of unique tokens
    Returns
    -------
        tokens: list
            list of unique tokens/SMILES alphabet.
        token2idx: dict
            dictionary mapping token to its index.
        num_tokens: int
            number of unique tokens.
    """
    if tokens is None:
        tokens = list(set(''.join(smiles)))
        tokens = list(np.sort(tokens))
        tokens = ''.join(tokens)
    token2idx = dict((token, i) for i, token in enumerate(tokens))
    num_tokens = len(tokens)
    return tokens, token2idx, num_tokens


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)



class VanillaQSAR(object):
    def __init__(self, model_instance=None, model_params=None,
                 model_type='classifier', ensemble_size=5, normalization=False):
        super(VanillaQSAR, self).__init__()
        self.model_instance = model_instance
        self.model_params = model_params
        self.ensemble_size = ensemble_size
        self.model = []
        self.normalization = normalization
        if model_type not in ['classifier', 'regressor']:
            raise InvalidArgumentError("model type must be either"
                                       "classifier or regressor")
        self.model_type = model_type
        if isinstance(self.model_instance, list):
            assert(len(self.model_instance) == self.ensemble_size)
            assert(isinstance(self.model_params, list))
            assert(len(self.model_params) == self.ensemble_size)
            for i in range(self.ensemble_size):
                self.model.append(self.model_instance[i](**model_params[i]))
        else:
            for _ in range(self.ensemble_size):
                self.model.append(self.model_instance(**model_params))
        if self.normalization:
            self.desc_mean = [0]*self.ensemble_size
        self.metrics_type = None

    def fit_model(self, data, cv_split='stratified'):
        eval_metrics = []
        x = data.x
        if self.model_type == 'classifier' and data.binary_y is not None:
            y = data.binary_y
        else:
            y = data.y
        cross_val_data, cross_val_labels = cross_validation_split(x=x, y=y,
                                                                  split=cv_split,
                                                                  n_folds=self.ensemble_size)
        for i in range(self.ensemble_size):
            train_x = np.concatenate(cross_val_data[:i] +
                                     cross_val_data[(i + 1):])
            test_x = cross_val_data[i]
            train_y = np.concatenate(cross_val_labels[:i] +
                                     cross_val_labels[(i + 1):])
            test_y = cross_val_labels[i]
            if self.normalization:
                train_x, desc_mean = normalize_desc(train_x)
                self.desc_mean[i] = desc_mean
                test_x, _ = normalize_desc(test_x, desc_mean)
            self.model[i].fit(train_x, train_y.ravel())
            predicted = self.model[i].predict(test_x)
            if self.model_type == 'classifier':
                eval_metrics.append(metrics.f1_score(test_y, predicted))
                self.metrics_type = 'F1 score'
            elif self.model_type == 'regressor':
                r2 = metrics.r2_score(test_y, predicted)
                eval_metrics.append(r2)
                self.metrics_type = 'R^2 score'
            else:
                raise RuntimeError()
        return eval_metrics, self.metrics_type

    def load_model(self, path):
        # TODO: add iterable path object instead of static path 
        self.model = joblib.load(path)
        if self.normalization:
            arr = np.load(path + 'desc_mean.npy')
            self.desc_mean = arr

    def save_model(self, path):
        joblib.dump(self.model, path + '.joblib')
        if self.normalization:
            np.save(path + 'desc_mean.npy', self.desc_mean)

    def predict(self, objects=None, average=True, get_features=None,
                **kwargs):
        objects = np.array(objects)
        invalid_objects = []
        processed_objects = []
        if get_features is not None:
            x, processed_indices, invalid_indices = get_features(objects,
                                                                 **kwargs)
            processed_objects = objects[processed_indices]
            invalid_objects = objects[invalid_indices]
        else:
            x = objects
        if len(x) == 0:
            processed_objects = []
            prediction = []
            invalid_objects = objects
        else:
            prediction = []
            for i in range(self.ensemble_size):
                m = self.model[i]
                if self.normalization:
                    x, _ = normalize_desc(x, self.desc_mean[i])
                prediction.append(m.predict(x))
            prediction = np.array(prediction)
            if average:
                prediction = prediction.mean(axis=0)
        return processed_objects, prediction, invalid_objects

class StackAugmentedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_type='GRU',
                 n_layers=1, is_bidirectional=False, has_stack=False,
                 stack_width=None, stack_depth=None, use_cuda=None,
                 optimizer_instance=torch.optim.Adadelta, lr=0.01):
        """
        Constructor for the StackAugmentedRNN object.
        Parameters
        ----------
        input_size: int
            number of characters in the alphabet
        hidden_size: int
            size of the RNN layer(s)
        output_size: int
            again number of characters in the alphabet
        layer_type: str (default 'GRU')
            type of the RNN layer to be used. Could be either 'LSTM' or 'GRU'.
        n_layers: int (default 1)
            number of RNN layers
        is_bidirectional: bool (default False)
            parameter specifying if RNN is bidirectional
        has_stack: bool (default False)
            parameter specifying if augmented memory stack is used
        stack_width: int (default None)
            if has_stack is True then this parameter defines width of the
            augmented stack memory
        stack_depth: int (default None)
            if has_stack is True then this parameter define depth of the augmented
            stack memory. Hint: no need fo stack depth to be larger than the
            length of the longest sequence you plan to generate
        use_cuda: bool (default None)
            parameter specifying if GPU is used for computations. If left
            unspecified, GPU will be used if available
        optimizer_instance: torch.optim object (default torch.optim.Adadelta)
            optimizer to be used for training
        lr: float (default 0.01)
            learning rate for the optimizer
        """
        super(StackAugmentedRNN, self).__init__()
        
        if layer_type not in ['GRU', 'LSTM']:
            raise InvalidArgumentError('Layer type must be GRU or LSTM')
        self.layer_type = layer_type
        self.is_bidirectional = is_bidirectional
        if self.is_bidirectional:
            self.num_dir = 2
        else:
            self.num_dir = 1
        if layer_type == 'LSTM':
            self.has_cell = True
        else:
            self.has_cell = False
        self.has_stack = has_stack
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if self.has_stack:
            self.stack_width = stack_width
            self.stack_depth = stack_depth

        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()

        self.n_layers = n_layers
        
        if self.has_stack:
            self.stack_controls_layer = nn.Linear(in_features=self.hidden_size *
                                                              self.num_dir,
                                                  out_features=3)

            self.stack_input_layer = nn.Linear(in_features=self.hidden_size *
                                                           self.num_dir,
                                               out_features=self.stack_width)

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.has_stack:
            rnn_input_size = hidden_size + stack_width
        else:
            rnn_input_size = hidden_size
        if self.layer_type == 'LSTM':
            self.rnn = nn.LSTM(rnn_input_size, hidden_size, n_layers,
                               bidirectional=self.is_bidirectional)
            self.decoder = nn.Linear(hidden_size * self.num_dir, output_size)
        elif self.layer_type == 'GRU':
            self.rnn = nn.GRU(rnn_input_size, hidden_size, n_layers,
                             bidirectional=self.is_bidirectional)
            self.decoder = nn.Linear(hidden_size * self.num_dir, output_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        
        if self.use_cuda:
            self = self.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer_instance = optimizer_instance
        self.optimizer = self.optimizer_instance(self.parameters(), lr=lr,
                                                 weight_decay=0.00001)
  
    def load_model(self, path):
        """
        Loads pretrained parameters from the checkpoint into the model.
        Parameters
        ----------
        path: str
            path to the checkpoint file model will be loaded from.
        """
        weights = torch.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(weights)

    def save_model(self, path):
        """
        Saves model parameters into the checkpoint file.
        Parameters
        ----------
        path: str
            path to the checkpoint file model will be saved to.
        """
        torch.save(self.state_dict(), path)

    def change_lr(self, new_lr):
        """
        Updates learning rate of the optimizer.
        Parameters
        ----------
        new_lr: float
            new learning rate value
        """
        self.optimizer = self.optimizer_instance(self.parameters(), lr=new_lr)
        self.lr = new_lr

    def forward(self, inp, hidden, stack):
        """
        Forward step of the model. Generates probability of the next character
        given the prefix.
        Parameters
        ----------
        inp: torch.tensor
            input tensor that contains prefix string indices
        hidden: torch.tensor or tuple(torch.tensor, torch.tensor)
            previous hidden state of the model. If layer_type is 'LSTM',
            then hidden is a tuple of hidden state and cell state, otherwise
            hidden is torch.tensor
        stack: torch.tensor
            previous state of the augmented memory stack
        Returns
        -------
        output: torch.tensor
            tensor with non-normalized probabilities of the next character
        next_hidden: torch.tensor or tuple(torch.tensor, torch.tensor)
            next hidden state of the model. If layer_type is 'LSTM',
            then next_hidden is a tuple of hidden state and cell state,
            otherwise next_hidden is torch.tensor
        next_stack: torch.tensor
            next state of the augmented memory stack
        """
        inp = self.encoder(inp.view(1, -1))
        if self.has_stack:
            if self.has_cell:
                hidden_ = hidden[0]
            else:
                hidden_ = hidden
            if self.is_bidirectional:
                hidden_2_stack = torch.cat((hidden_[0], hidden_[1]), dim=1)
            else:
                hidden_2_stack = hidden_.squeeze(0)
            stack_controls = self.stack_controls_layer(hidden_2_stack)
            stack_controls = F.softmax(stack_controls, dim=1)
            stack_input = self.stack_input_layer(hidden_2_stack.unsqueeze(0))
            stack_input = torch.tanh(stack_input)
            stack = self.stack_augmentation(stack_input.permute(1, 0, 2),
                                            stack, stack_controls)
            stack_top = stack[:, 0, :].unsqueeze(0)
            inp = torch.cat((inp, stack_top), dim=2)
        output, next_hidden = self.rnn(inp.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, next_hidden, stack

    def stack_augmentation(self, input_val, prev_stack, controls):
        """
        Augmentation of the tensor into the stack. For more details see
        https://arxiv.org/abs/1503.01007
        Parameters
        ----------
        input_val: torch.tensor
            tensor to be added to stack
        prev_stack: torch.tensor
            previous stack state
        controls: torch.tensor
            predicted probabilities for each operation in the stack, i.e
            PUSH, POP and NO_OP. Again, see https://arxiv.org/abs/1503.01007
        Returns
        -------
        new_stack: torch.tensor
            new stack state
        """
        batch_size = prev_stack.size(0)

        controls = controls.view(-1, 3, 1, 1)
        zeros_at_the_bottom = torch.zeros(batch_size, 1, self.stack_width)
        if self.use_cuda:
            zeros_at_the_bottom = Variable(zeros_at_the_bottom.cuda())
        else:
            zeros_at_the_bottom = Variable(zeros_at_the_bottom)
        a_push, a_pop, a_no_op = controls[:, 0], controls[:, 1], controls[:, 2]
        stack_down = torch.cat((prev_stack[:, 1:], zeros_at_the_bottom), dim=1)
        stack_up = torch.cat((input_val, prev_stack[:, :-1]), dim=1)
        new_stack = a_no_op * prev_stack + a_push * stack_up + a_pop * stack_down
        return new_stack

    def init_hidden(self):
        """
        Initialization of the hidden state of RNN.
        Returns
        -------
        hidden: torch.tensor
            tensor filled with zeros of an appropriate size (taking into
            account number of RNN layers and directions)
        """
        if self.use_cuda:
            return Variable(torch.zeros(self.n_layers * self.num_dir, 1,
                                        self.hidden_size).cuda())
        else:
            return Variable(torch.zeros(self.n_layers * self.num_dir, 1,
                                        self.hidden_size))

    def init_cell(self):
        """
        Initialization of the cell state of LSTM. Only used when layers_type is
        'LSTM'
        Returns
        -------
        cell: torch.tensor
            tensor filled with zeros of an appropriate size (taking into
            account number of RNN layers and directions)
        """
        if self.use_cuda:
            return Variable(torch.zeros(self.n_layers * self.num_dir, 1,
                                        self.hidden_size).cuda())
        else:
            return Variable(torch.zeros(self.n_layers * self.num_dir, 1,
                                        self.hidden_size))

    def init_stack(self):
        """
        Initialization of the stack state. Only used when has_stack is True
        Returns
        -------
        stack: torch.tensor
            tensor filled with zeros
        """
        result = torch.zeros(1, self.stack_depth, self.stack_width)
        if self.use_cuda:
            return Variable(result.cuda())
        else:
            return Variable(result)

    def train_step(self, inp, target):
        """
        One train step, i.e. forward-backward and parameters update, for
        a single training example.
        Parameters
        ----------
        inp: torch.tensor
            tokenized training string from position 0 to position (seq_len - 1)
        target:
            tokenized training string from position 1 to position seq_len
        Returns
        -------
        loss: float
            mean value of the loss function (averaged through the sequence
            length)
        """
        hidden = self.init_hidden()
        if self.has_cell:
            cell = self.init_cell()
            hidden = (hidden, cell)
        if self.has_stack:
            stack = self.init_stack()
        else:
            stack = None
        self.optimizer.zero_grad()
        loss = 0
        for c in range(len(inp)):
            output, hidden, stack = self(inp[c], hidden, stack)
            loss += self.criterion(output, target[c].unsqueeze(0))

        loss.backward()
        self.optimizer.step()

        return loss.item() / len(inp)
    
    def evaluate(self, data, prime_str='<', end_token='>', predict_len=100):
        """
        Generates new string from the model distribution.
        Parameters
        ----------
        data: object of type GeneratorData
            stores information about the generator data format such alphabet, etc
        prime_str: str (default '<')
            prime string that will be used as prefix. Deafult value is just the
            START_TOKEN
        end_token: str (default '>')
            when end_token is sampled from the model distribution,
            the generation of a new example is finished
        predict_len: int (default 100)
            maximum length of the string to be generated. If the end_token is
            not sampled, the generation will be aborted when the length of the
            generated sequence is equal to predict_len
        Returns
        -------
        new_sample: str
            Newly generated sample from the model distribution.
        """
        hidden = self.init_hidden()
        if self.has_cell:
            cell = self.init_cell()
            hidden = (hidden, cell)
        if self.has_stack:
            stack = self.init_stack()
        else:
            stack = None
        prime_input = data.char_tensor(prime_str)
        new_sample = prime_str

        # Use priming string to "build up" hidden state
        for p in range(len(prime_str)-1):
            _, hidden, stack = self.forward(prime_input[p], hidden, stack)
        inp = prime_input[-1]

        for p in range(predict_len):
            output, hidden, stack = self.forward(inp, hidden, stack)

            # Sample from the network as a multinomial distribution
            probs = torch.softmax(output, dim=1)
            top_i = torch.multinomial(probs.view(-1), 1)[0].cpu().numpy()

            # Add predicted character to string and use as next input
            predicted_char = data.all_characters[top_i]
            new_sample += predicted_char
            inp = data.char_tensor(predicted_char)
            if predicted_char == end_token:
                break

        return new_sample

    def fit(self, data, n_iterations, all_losses=[], print_every=100,
            plot_every=10, augment=False):
        """
        This methods fits the parameters of the model. Training is performed to
        minimize the cross-entropy loss when predicting the next character
        given the prefix.
        Parameters
        ----------
        data: object of type GeneratorData
            stores information about the generator data format such alphabet, etc
        n_iterations: int
            how many iterations of training will be performed
        all_losses: list (default [])
            list to store the values of the loss function
        print_every: int (default 100)
            feedback will be printed to std_out once every print_every
            iterations of training
        plot_every: int (default 10)
            value of the loss function will be appended to all_losses once every
            plot_every iterations of training
        augment: bool (default False)
            parameter specifying if SMILES enumeration will be used. For mode
            details on SMILES enumeration see https://arxiv.org/abs/1703.07076
        Returns
        -------
        all_losses: list
            list that stores the values of the loss function (learning curve)
        """
        start = time.time()
        loss_avg = 0

        if augment:
            smiles_augmentation = SmilesEnumerator()
        else:
            smiles_augmentation = None

        for epoch in trange(1, n_iterations + 1, desc='Training in progress...'):
            inp, target = data.random_training_set(smiles_augmentation)
            loss = self.train_step(inp, target)
            loss_avg += loss

            if epoch % print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch,
                                               epoch / n_iterations * 100, loss)
                      )
                print(self.evaluate(data=data, prime_str = '<',
                                    predict_len=100), '\n')

            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0
        return all_losses

class SmilesEnumerator(object):
    """SMILES Enumerator, vectorizer and devectorizer
    #Arguments
        charset: string containing the characters for the vectorization
          can also be generated via the .fit() method
        pad: Length of the vectorization
        leftpad: Add spaces to the left of the SMILES
        isomericSmiles: Generate SMILES containing information about stereogenic centers
        enum: Enumerate the SMILES during transform
        canonical: use canonical SMILES during transform (overrides enum)
    """

    def __init__(self, charset='@C)(=cOn1S2/H[N]\\', pad=120, leftpad=True, isomericSmiles=True, enum=True,
                 canonical=False):
        self._charset = None
        self.charset = charset
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical

    @property
    def charset(self):
        return self._charset

    @charset.setter
    def charset(self, charset):
        self._charset = charset
        self._charlen = len(charset)
        self._char_to_int = dict((c, i) for i, c in enumerate(charset))
        self._int_to_char = dict((i, c) for i, c in enumerate(charset))

    def fit(self, smiles, extra_chars=[], extra_pad=5):
        """Performs extraction of the charset and length of a SMILES datasets and sets self.pad and self.charset
        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
            extra_chars: List of extra chars to add to the charset (e.g. "\\\\" when "/" is present)
            extra_pad: Extra padding to add before or after the SMILES vectorization
        """
        charset = set("".join(list(smiles)))
        self.charset = "".join(charset.union(set(extra_chars)))
        self.pad = max([len(smile) for smile in smiles]) + extra_pad

    def randomize_smiles(self, smiles):
        """Perform a randomization of a SMILES string
        must be RDKit sanitizable"""
        m = Chem.MolFromSmiles(smiles)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

    def transform(self, smiles):
        """Perform an enumeration (randomization) and vectorization of a Numpy array of smiles strings
        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
        """
        one_hot = np.zeros((smiles.shape[0], self.pad, self._charlen), dtype=np.int8)

        for i, ss in enumerate(smiles):
            if self.enumerate: ss = self.randomize_smiles(ss)
            for j, c in enumerate(ss):
                one_hot[i, j, self._char_to_int[c]] = 1
        return one_hot

    def reverse_transform(self, vect):
        """ Performs a conversion of a vectorized SMILES to a smiles strings
        charset must be the same as used for vectorization.
        #Arguments
            vect: Numpy array of vectorized SMILES.
        """
        smiles = []
        for v in vect:
            # mask v
            v = v[v.sum(axis=1) == 1]
            # Find one hot encoded index with argmax, translate to char and join to string
            smile = "".join(self._int_to_char[i] for i in v.argmax(axis=1))
            smiles.append(smile)
        return np.array(smiles)

def cross_validation_split(x, y, n_folds=5, split='random', folds=None):
    assert(len(x) == len(y))
    x = np.array(x)
    y = np.array(y)
    if split not in ['random', 'stratified', 'fixed']:
        raise ValueError('Invalid value for argument \'split\': '
                         'must be either \'random\', \'stratified\' '
                         'or \'fixed\'')
    if split == 'random':
        cv_split = KFold(n_splits=n_folds, shuffle=True)
        folds = list(cv_split.split(x, y))
    elif split == 'stratified':
        cv_split = StratifiedKFold(n_splits=n_folds, shuffle=True)
        folds = list(cv_split.split(x, y))
    elif split == 'fixed' and folds is None:
        raise TypeError(
            'Invalid type for argument \'folds\': found None, but must be list')
    cross_val_data = []
    cross_val_labels = []
    if len(folds) == n_folds:
        for fold in folds:
            cross_val_data.append(x[fold[1]])
            cross_val_labels.append(y[fold[1]])
    elif len(folds) == len(x) and np.max(folds) == n_folds:
        for f in range(n_folds):
            left = np.where(folds == f)[0].min()
            right = np.where(folds == f)[0].max()
            cross_val_data.append(x[left:right + 1])
            cross_val_labels.append(y[left:right + 1])

    return cross_val_data, cross_val_labels

class PredictorData(object):
    def __init__(self, path, delimiter=',', cols=[0, 1], get_features=None,
                 has_label=True, labels_start=1, **kwargs):
        super(PredictorData, self).__init__()
        data = read_object_property_file(path, delimiter, cols_to_read=cols)
        if has_label:
            self.objects = np.array(data[:labels_start]).reshape(-1)
            self.y = np.array(data[labels_start:], dtype='float32')
            self.y = self.y.reshape(-1, len(cols) - labels_start)
            if self.y.shape[1] == 1:
                self.y = self.y.reshape(-1)
        else:
            self.objects = np.array(data[:labels_start]).reshape(-1)
            self.y = [None]*len(self.objects)
        assert len(self.objects) == len(self.y)
        if get_features is not None:
            self.x, processed_indices, invalid_indices = \
                get_features(self.objects, **kwargs)
            self.invalid_objects = self.objects[invalid_indices]
            self.objects = self.objects[processed_indices]
            self.invalid_y = self.y[invalid_indices]
            self.y = self.y[processed_indices]
        else:
            self.x = self.objects
            self.invalid_objects = None
            self.invalid_y = None
        self.binary_y = None
    def binarize(self, threshold):
        self.binary_y = np.array(self.y >= threshold, dtype='int32')

class GeneratorData(object):
    def __init__(self, training_data_path, tokens=None, start_token='<', 
                 end_token='>', max_len=120, use_cuda=None, **kwargs):

        super(GeneratorData, self).__init__()

        if 'cols_to_read' not in kwargs:
            kwargs['cols_to_read'] = []

        data = read_object_property_file(training_data_path,
                                                       **kwargs)
        self.start_token = start_token
        self.end_token = end_token
        self.file = []
        for i in range(len(data)):
            if len(data[i]) <= max_len:
                self.file.append(self.start_token + data[i] + self.end_token) 
        self.file_len = len(self.file)
        self.all_characters, self.char2idx, \
        self.n_characters = tokenize(self.file, tokens)
        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()

    def load_dictionary(self, tokens, char2idx):
        self.all_characters = tokens
        self.char2idx = char2idx
        self.n_characters = len(tokens)

    def random_chunk(self):

        index = random.randint(0, self.file_len-1)
        return self.file[index]

    def char_tensor(self, string):

        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = self.all_characters.index(string[c])
        if self.use_cuda:
            return torch.tensor(tensor).cuda()
        else:
            return torch.tensor(tensor)

    def random_training_set(self, smiles_augmentation):
        chunk = self.random_chunk()
        if smiles_augmentation is not None:
            chunk = '<' + smiles_augmentation.randomize_smiles(chunk[1:-1]) + '>'
        inp = self.char_tensor(chunk[:-1])
        target = self.char_tensor(chunk[1:])
        return inp, target

    def read_sdf_file(self, path, fields_to_read):
        raise NotImplementedError
        
    def update_data(self, path):
        self.file, success = read_smi_file(path, unique=True)
        self.file_len = len(self.file)
        assert success

def read_object_property_file(path, delimiter=',', cols_to_read=[0, 1],
                              keep_header=False):
    f = open(path, 'r')
    reader = csv.reader(f, delimiter=delimiter)
    data_full = np.array(list(reader))
    if keep_header:
        start_position = 0
    else:
        start_position = 1
    assert len(data_full) > start_position
    data = [[] for _ in range(len(cols_to_read))]
    for i in range(len(cols_to_read)):
        col = cols_to_read[i]
        data[i] = data_full[start_position:, col]
    f.close()
    if len(cols_to_read) == 1:
        data = data[0]
    return data


def estimate_and_update(generator, predictor, n_to_generate, **kwargs):
    generated = []
    pbar = tqdm(range(n_to_generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        generated.append(generator.evaluate(gen_data, predict_len=120)[1:-1])
    sanitized = canonical_smiles(generated, sanitize=False, throw_warning=False)[:-1]
    unique_smiles = list(np.unique(sanitized))[1:]
    smiles, prediction, nan_smiles = predictor.predict(unique_smiles, get_features=get_fp)
    return smiles, prediction

def get_fp(smiles):
    fp = []
    processed_indices = []
    invalid_indices = []
    for i in range(len(smiles)):
        mol = smiles[i]
        tmp = np.array(mol2image(mol, n=2048))
        if np.isnan(tmp[0]):
            invalid_indices.append(i)
        else:
            fp.append(tmp)
            processed_indices.append(i)
    return np.array(fp), processed_indices, invalid_indices

def mol2image(x, n=2048):
    try:
        m = Chem.MolFromSmiles(x)
        fp = Chem.RDKFingerprint(m, maxPath=4, fpSize=n)
        res = np.zeros(len(fp))
        DataStructs.ConvertToNumpyArray(fp, res)
        return res
    except:
        return [np.nan]

def init():
    global use_cuda
    global my_generator
    global gen_data
    global my_predictor
    
    hidden_size = 1500
    stack_width = 1500
    stack_depth = 200
    layer_type = 'GRU'
    n_characters = 45
    lr = 0.001
    optimizer_instance = torch.optim.Adadelta
    # model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), './deploy_files')
    use_cuda = False
    # gen_data_path = model_path +'/1000smiles.csv' 
    gen_data_path = './deploy_files/1000smiles.csv'
    tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']
    
    gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t', 
                         cols_to_read=[0], keep_header=True, tokens=tokens)

    my_generator = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=hidden_size,
                                 output_size=gen_data.n_characters, layer_type=layer_type,
                                 n_layers=1, is_bidirectional=False, has_stack=True,
                                 stack_width=stack_width, stack_depth=stack_depth, 
                                 use_cuda=use_cuda, 
                                 optimizer_instance=optimizer_instance, lr=lr)

    # gen_model_path = model_path +'/generative_model_max.pth'
    gen_model_path = "./deploy_files/generative_model_max.pth"

    my_generator.load_model(gen_model_path)

    pred_data = PredictorData(path='./deploy_files/jak2_data.csv', get_features=get_fp)

    #my_predict = predictor model 
    model_instance = RFR
    model_params = {'n_estimators': 175, 'n_jobs': 10}  

    my_predictor = VanillaQSAR(model_instance=model_instance,
                              model_params=model_params,
                              model_type='regressor')
    
    my_predictor.fit_model(pred_data, cv_split='random')
    my_predictor.save_model('./deploy_files/vanillaqsar')
    my_predictor.load_model('./deploy_files/vanillaqsar.joblib')

# @input_schema('n_to_generate', NumpyParameterType(input_sample))
# @output_schema(NumpyParameterType(output_sample))
def run(n_to_generate):
    try:
        smiles, pic50 = estimate_and_update(my_generator,my_predictor,n_to_generate=n_to_generate)

        molecules = [Chem.MolFromSmiles(x) for x in smiles]
        qed_list = [] 
        for x in molecules:
            try:
                qed_list.append(QED.qed(x))
            except Exception as e:
                print("early error")
                print(e)
                # pass
        # return smiles_biased_max.tolist()
        return smiles.tolist(), pic50.tolist(), qed_list
    except Exception as e:
        print("oof error time")
        error = str(e)
        return error
    
        
if __name__ == "__main__":
    init()
    print(run(5))