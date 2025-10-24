import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple, Optional
from torch import Tensor
import numbers
import config_file as c
import numpy as np

from collections import OrderedDict


class RNNetwork(nn.Module):
    '''
    Recurrent neural network class
    '''

    def __init__(self, input_size, hidden_size, num_layers, nonlinearity, bias, batch_first, dropout, norm, ind_rnn, weight_constraint):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.ind_rnn = ind_rnn
        self.binary_output = c.BINARY_OUTPUT

        # this is inheriting from torch.nn.RNN
        # cRNN contains a ModuleList with all layers
        # each layer is a Module itself
        self.rnn = cRNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                        nonlinearity=nonlinearity, bias=bias, batch_first=batch_first, dropout=dropout,
                        norm=norm, ind_rnn=ind_rnn, weight_constraint=weight_constraint)

        if c.BINARY_OUTPUT == True:
            if isinstance(hidden_size,list):
                self.out = nn.Sequential(
                    nn.Linear(hidden_size[-1], input_size), 
                    nn.Sigmoid()
                    )
            else:
                self.out = nn.Sequential(
                    nn.Linear(hidden_size, input_size), 
                    nn.Sigmoid()
                    )
        else:
            if isinstance(hidden_size,list):
                self.out = nn.Linear(hidden_size[-1], input_size)
            else:
                self.out = nn.Linear(hidden_size, input_size)


    def forward(self, input: Tensor, hstates: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        last_h_t, h_n = self.rnn(input, hstates)

        # Linear layer
        # out layer is calculated for each element of sequence
        out = self.out(last_h_t) 
        
        return out, h_n


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias):
        super(RNNCell, self).__init__()
        self.bias = bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # He initialization
        self.weight_ih = Parameter(torch.normal(mean = torch.from_numpy(np.array([np.repeat(0,hidden_size)]* input_size).astype(np.float32)), std = torch.from_numpy(np.array([np.repeat(np.sqrt(2/input_size),hidden_size)]* input_size).astype(np.float32)))) 
        self.weight_hh = Parameter(torch.normal(mean = torch.from_numpy(np.array([np.repeat(0,hidden_size)]* hidden_size).astype(np.float32)), std = torch.from_numpy(np.array([np.repeat(np.sqrt(2/input_size),hidden_size)]* hidden_size).astype(np.float32))))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(np.repeat(0,hidden_size).astype(np.float32)))
            self.bias_hh = Parameter(torch.Tensor(np.repeat(0,hidden_size).astype(np.float32)))

    def forward(self, input: Tensor, hstate: Optional[Tensor] = None) -> Tensor:
        assert input.dim() in (1, 2), \
            f"RNNCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        # this statement assumes batch first because input.size(0) is bs and not input_size
        if hstate is None:
            hstate = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hstate = hstate.unsqueeze(0) if not is_batched else hstate

        # inputs * weights
        igate = torch.mm(input, self.weight_ih)
        hgate = torch.mm(hstate, self.weight_hh)

        if self.bias:
            igate = igate + self.bias_ih
            hgate = hgate + self.bias_hh

        # new hstate
        gate = igate + hgate

        if not is_batched:
            gate = gate.squeeze(0)

        return gate



class IndRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias, weight_constraint=False):
        super(IndRNNCell, self).__init__()
        self.bias = bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # He initialization
        if c.NONLINEARITY == "relu":
            self.weight_ih = Parameter(torch.normal(mean = torch.from_numpy(np.array([np.repeat(0,hidden_size)]* input_size).astype(np.float32)), std = torch.from_numpy(np.array([np.repeat(np.sqrt(2/input_size),hidden_size)]* input_size).astype(np.float32))))
        if weight_constraint == True and c.NONLINEARITY == "relu":
            self.weight_hh = Parameter(c.RWEIGHT_CONSTRAINT * torch.normal(mean = torch.from_numpy(np.repeat(0,hidden_size).astype(np.float32)), std = torch.from_numpy(np.repeat(np.sqrt(2/input_size),hidden_size).astype(np.float32))))  
        else:
            if c.NONLINEARITY == "relu":
                self.weight_hh = Parameter(torch.normal(mean = torch.from_numpy(np.repeat(0,hidden_size).astype(np.float32)), std = torch.from_numpy(np.repeat(np.sqrt(2/input_size),hidden_size).astype(np.float32))))
            
        if bias:
            if c.NONLINEARITY == "relu":
                self.bias_ih = Parameter(torch.Tensor(np.repeat(0,hidden_size).astype(np.float32)))
                self.bias_hh = Parameter(torch.Tensor(np.repeat(0,hidden_size).astype(np.float32)))
                
                

    def forward(self, input: Tensor, hstate: Optional[Tensor] = None) -> Tensor:
        assert input.dim() in (1, 2), \
            f"RNNCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        # this statement assumes batch first because input.size(0) is bs and not input_size
        # size of the input here is batch size --> this is because all time steps (frames) are inputted one by one
        # but all of the movies can still be done at once (just for efficiency)
        # time steps are done one by one so the hstates can be updated with each time step
        if hstate is None:
            # here for all first time steps (because batch_movies are inputted into the network with hstates None)
            # get a 0 activation of the initial hidden state
            hstate = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hstate = hstate.unsqueeze(0) if not is_batched else hstate

        # inputs * weights
        igate = torch.mm(input, self.weight_ih)
        # NOT mm because this is an indRNN (Li et al., 2018) 
        # --> nodes are not recurrent to other nodes in the layer, only to themselves
        hgate = torch.mul(hstate, self.weight_hh)

        if self.bias:
            igate = igate + self.bias_ih
            hgate = hgate + self.bias_hh

        # new hstate
        gate = igate + hgate

        if not is_batched:
            gate = gate.squeeze(0)

        return gate


class RNNLayer(nn.Module):
    """
    Describes a single layer of a custom RNN. If RNN cell does not contain activation. It has to be done here.
    If every layer has a convolutional filtering, then this has to be done here.
    """

    def __init__(self, input_size, hidden_size, nonlinearity, bias, batch_first, norm, ind_rnn, weight_constraint):
        super(RNNLayer, self).__init__()
        self.seq_idx = 1 if batch_first else 0
        self.batch_first = batch_first
        self.nonlinearity = nonlinearity

        if norm == "layer_norm":
            self.norm = nn.LayerNorm(hidden_size)
            
        # batch norm makes less sense in RNN --> we stick with layer_norm
        elif norm == "batch_norm":
            self.norm = nn.BatchNorm1d(hidden_size) # a boolean value that when set to True, this module has learnable affine parameters. Default: True
       # elif norm == "instance_norm":
            #self.norm = nn.InstanceNorm1d(hidden_size) # implemention did something unexpected (also wasn't conceptually what we thought made most sense)
        else:
            self.norm = None
            bias = True

        if ind_rnn == True:
            self.rnn_cell = IndRNNCell(input_size, hidden_size, bias, weight_constraint)
        else:
            self.rnn_cell = RNNCell(input_size, hidden_size, bias)

    def forward(self, input: Tensor, hstate: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        # here movie frames are one by one fed through the network
        inputs = input.unbind(self.seq_idx)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            hstate = self.rnn_cell(inputs[i], hstate)
            if self.norm:
                hstate = self.norm(hstate)
            if self.nonlinearity == "relu":
                hstate = torch.relu(hstate)
            elif self.nonlinearity == "relu_with_max":
                hstate = torch.clamp(hstate, min=0, max=1)
            elif self.nonlinearity == "tanh":
                hstate = torch.tanh(hstate)
            elif self.nonlinearity == "sigmoid":
                hstate = torch.sigmoid(hstate)
            else:
                raise RuntimeError(
                    "Unknown nonlinearity: {}".format(self.nonlinearity))

            outputs += [hstate]
        outputs = torch.stack(outputs, dim=self.seq_idx)

        return outputs, hstate


def init_rnn_layers(input_size, hidden_size, num_layers, nonlinearity, bias, batch_first, norm, ind_rnn, weight_constraint):

    if isinstance(hidden_size, list):
        layers = [RNNLayer(input_size, hidden_size[0], nonlinearity, bias, batch_first, norm, ind_rnn, weight_constraint)] + \
                 [RNNLayer(hidden_size[l], hidden_size[l+1], nonlinearity, bias, batch_first, norm, ind_rnn, weight_constraint)
                                                               for l in range(num_layers - 1)]
    else:
        
        layers = [RNNLayer(input_size, hidden_size, nonlinearity, bias, batch_first, norm, ind_rnn, weight_constraint)] + \
                 [RNNLayer(hidden_size, hidden_size, nonlinearity, bias, batch_first, norm, ind_rnn, weight_constraint)
                                                               for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)




class cRNN(nn.Module):
    '''
    collection of RNN layers
    '''

    def __init__(self, input_size, hidden_size, num_layers, nonlinearity, bias, batch_first, dropout, norm, ind_rnn, weight_constraint):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.layers = init_rnn_layers(input_size, hidden_size, num_layers, nonlinearity, bias, batch_first, norm, ind_rnn, weight_constraint)

        if num_layers == 1:
            warnings.warn("dropout rnn adds dropout num_layers after all but last "
                          "recurrent layer, it expects num_layers greater than "
                          "1, but got num_layers = 1")

        # Dropout has to be here because last layer cannot have dropout
        # the last layer, with dropout probability = dropout.
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input: Tensor, hstates: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        output_hstates = jit.annotate(List[Tensor], [])
        output = input
        i = 0
        for rnn_layer in self.layers:
            hstate = hstates[i] if hstates is not None else hstates
            output, _ = rnn_layer(output, hstate)
            # Apply the dropout layer except the last layer
            if i < self.num_layers - 1:
                output = self.dropout_layer(output)

            # save all outputs, last hstate is last element from outputs
            output_hstates += [output]
            i += 1
            
        
            
        if isinstance(self.hidden_size, list):
            # make all tensors the same size so they can be stacked easily 
            largest_tensor_idx = [hs.shape for hs in output_hstates].index(max([hs.shape for hs in output_hstates]))
            
            for layer_id in range(len(self.layers)):
                nan_torch = torch.full((output_hstates[layer_id].shape[0], output_hstates[layer_id].shape[1], output_hstates[largest_tensor_idx].shape[2] - output_hstates[layer_id].shape[2]), torch.nan)
                output_hstates[layer_id] = torch.cat((output_hstates[layer_id],nan_torch),dim=2)
            
            
        return output, torch.stack(output_hstates)



class NNetwork(nn.Module):
    '''
    Neural network without recurrency
    '''

    def __init__(self, input_size, hidden_size, num_layers, nonlinearity, bias, batch_first, dropout, norm, *nargs, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.binary_output = c.BINARY_OUTPUT

        # this is inheriting from torch.nn.RNN
        # cNN contains a ModuleList with all layers
        # each layer is a Module itself
        self.nnet = cNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                        nonlinearity=nonlinearity, bias=bias, batch_first=batch_first, dropout=dropout,
                        norm=norm)

        if c.BINARY_OUTPUT == True:
            if isinstance(hidden_size,list):
                self.out = nn.Sequential(
                    nn.Linear(hidden_size[-1], input_size), 
                    nn.Sigmoid()
                    )
            else:
                self.out = nn.Sequential(
                    nn.Linear(hidden_size, input_size), 
                    nn.Sigmoid()
                    )
        else:
            if isinstance(hidden_size,list):
                self.out = nn.Linear(hidden_size[-1], input_size)
            else:
                self.out = nn.Linear(hidden_size, input_size)


    def forward(self, input: Tensor, hstates: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
     
        last_h_t, h_n = self.nnet(input, hstates)

        # Linear layer
        # out layer is calculated for each element of sequence
        out = self.out(last_h_t) 
        
        return out, h_n

class NNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias):
        super(NNCell, self).__init__()
        self.bias = bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # He initialization
        if c.NONLINEARITY == "relu":
            self.weight_ih = Parameter(torch.normal(mean = torch.from_numpy(np.array([np.repeat(0,hidden_size)]* input_size).astype(np.float32)), std = torch.from_numpy(np.array([np.repeat(np.sqrt(2/input_size),hidden_size)]* input_size).astype(np.float32))))
            
        if bias:
            if c.NONLINEARITY == "relu":
                self.bias_ih = Parameter(torch.Tensor(np.repeat(0,hidden_size).astype(np.float32)))

    def forward(self, input: Tensor, hstate: Optional[Tensor] = None) -> Tensor:
        assert input.dim() in (1, 2), \
            f"NNCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        # this statement assumes batch first because input.size(0) is bs and not input_size
        if hstate is None:
            hstate = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hstate = hstate.unsqueeze(0) if not is_batched else hstate

        igate = torch.mm(input, self.weight_ih)

        if self.bias:
            igate = igate + self.bias_ih

        gate = igate

        if not is_batched:
            gate = gate.squeeze(0)

        return gate


class NNLayer(nn.Module):
    """
    Describes a single layer of a custom NN. 
    If NN cell does not contain activation. It has to be done here.
    If every layer has a convolutional filtering, then this has to be done here.
    """

    def __init__(self, input_size, hidden_size, nonlinearity, bias, batch_first, norm):
        super(NNLayer, self).__init__()
        self.seq_idx = 1 if batch_first else 0
        self.batch_first = batch_first
        self.nonlinearity = nonlinearity

        if norm == "layer_norm":
            self.norm = nn.LayerNorm(hidden_size)
        elif norm == "batch_norm":
            self.norm = nn.BatchNorm1d(hidden_size) # a boolean value that when set to True, this module has learnable affine parameters. Default: True
       # elif norm == "instance_norm":
            #self.norm = nn.InstanceNorm1d(hidden_size) # a boolean value that when set to True, this module has learnable affine parameters. Default: False
        else:
            self.norm = None
            bias = True

        self.nn_cell = NNCell(input_size, hidden_size, bias)

    def forward(self, input: Tensor, hstate: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        # here movie frames are split and one by one fed through the network
        inputs = input.unbind(self.seq_idx)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            hstate = self.nn_cell(inputs[i], hstate)
            if self.norm:
                hstate = self.norm(hstate)
            if self.nonlinearity == "relu":
                hstate = torch.relu(hstate)
            elif self.nonlinearity == "relu_with_max":
                hstate = torch.clamp(hstate, min=0, max=1)
            elif self.nonlinearity == "tanh":
                hstate = torch.tanh(hstate)
            elif self.nonlinearity == "sigmoid":
                hstate = torch.sigmoid(hstate)
            else:
                raise RuntimeError(
                    "Unknown nonlinearity: {}".format(self.nonlinearity))

            outputs += [hstate]
        outputs = torch.stack(outputs, dim=self.seq_idx)

        return outputs, hstate


def init_nn_layers(input_size, hidden_size, num_layers, nonlinearity, bias, batch_first, norm):

    if isinstance(hidden_size, list):
        layers = [NNLayer(input_size, hidden_size[0], nonlinearity, bias, batch_first, norm)] + \
                 [NNLayer(hidden_size[l], hidden_size[l+1], nonlinearity, bias, batch_first, norm)
                                                               for l in range(num_layers - 1)]
    else:
        
        layers = [NNLayer(input_size, hidden_size, nonlinearity, bias, batch_first, norm)] + \
                 [NNLayer(hidden_size, hidden_size, nonlinearity, bias, batch_first, norm)
                                                               for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)




class cNN(nn.Module):
    '''
    collection of NN layers
    '''

    def __init__(self, input_size, hidden_size, num_layers, nonlinearity, bias, batch_first, dropout, norm):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.layers = init_nn_layers(input_size, hidden_size, num_layers, nonlinearity, bias, batch_first, norm)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input: Tensor, hstates: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        output_hstates = jit.annotate(List[Tensor], [])
        output = input
        i = 0
        for nn_layer in self.layers:
            hstate = hstates[i] if hstates is not None else hstates
            output, _ = nn_layer(output, hstate)
            if i < self.num_layers:
                output = self.dropout_layer(output)

            # save all outputs, last hstate is last element from outputs
            output_hstates += [output]
            i += 1
            
        
            
        if isinstance(self.hidden_size, list):
            # make all tensors the same size so they can be stacked easily 
            largest_tensor_idx = [hs.shape for hs in output_hstates].index(max([hs.shape for hs in output_hstates]))
            
            for layer_id in range(len(self.layers)):
                nan_torch = torch.full((output_hstates[layer_id].shape[0], output_hstates[layer_id].shape[1], output_hstates[largest_tensor_idx].shape[2] - output_hstates[layer_id].shape[2]), torch.nan)
                output_hstates[layer_id] = torch.cat((output_hstates[layer_id],nan_torch),dim=2)
            
            
        return output, torch.stack(output_hstates)

