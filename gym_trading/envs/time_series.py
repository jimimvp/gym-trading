import gym
import gym
from gym import spaces
import numpy as np
import pandas as pd
from collections import namedtuple
from gym_trading.envs.utils import Capital, Order, OrderType

import pandas as pd
import copy
import random
from itertools import tee
from gym_trading.envs.utils import hdf_generator
import os

from torch_rl.utils import RunningMeanStd, gauss_weights_init, to_tensor as tt
from torch_rl.models.core import NeuralNet

import torch as tor
from torch import nn

random.seed(123)

path = os.environ['BINANCE_DATA_PATH']



class RecurrentNet(NeuralNet):


    def __init__(self, architecture, recurrent_layers=2, weight_init=gauss_weights_init(0,0.02),activation_functions=None, bidirectional=True, recurr_type=nn.GRU):
        """ 
            First number of architecture indicates number of GRU units
        """
        super(RecurrentNet, self).__init__()
        if len(architecture) < 2:
            raise Exception("Architecture needs at least two numbers to create network")
        #assert architecture[-1]%2 == 1, "Last layer has to represent 2*actions_space for the Gaussian + 1 for value"
        self.activation_functions = activation_functions
        self.layer_list = []
        self.layer_list_val = []
        self.recurrent = True
 
        
        recurr_size = architecture[0]
        #architecture = architecture[1:]


        self.recurrent_unit = recurr_type(input_size=recurr_size, hidden_size=recurr_size, num_layers=recurrent_layers,batch_first=True, bidirectional=bidirectional)


        for i in range(len(architecture)-1):
            self.layer_list.append(nn.Linear(architecture[i], architecture[i+1]))
            setattr(self, "fc" + str(i), self.layer_list[-1])

        self.h_n = None

        self.apply(weight_init)



    def forward(self, x, hx=None):

        
        # Policy network

        x, h_n = self.recurrent_unit(x, hx)
        #print(h_n.data.numpy())
        #Take hidden layer
        self.h_n = h_n
        x = h_n[-1, :, :].squeeze()

        if self.activation_functions:
            for i, func in enumerate(self.activation_functions):
                x = func(self.layer_list[i](x))
        else:
            for i, layer in enumerate(self.layer_list[:-1]):
                x = self.tanh(layer(x))

        x = self.layer_list[-1](x)

        x = self.tanh(x)


        return x


    def reset(self):
        #Equivalent to setting hidden state to 0 in pytorch
        self.h_n = None

    def __call__(self, s, use_last_state=True):
        if use_last_state:
            hx = self.h_n
        return self.forward(s, hx)

from torch.optim import Adam



if __name__ == '__main__':
    
    print(path) 
    #hdf_generator(path, chunk_size=10000, pair=None, columns=None):
    generator = hdf_generator(path, pair='ETHBTC', columns=['p', 'P', 'x', 'c', 'b', 'a', 'o', 'h', 'l', 'v', 'n'])()
    mean_calc = None
    steps = 10
    data_next = next(generator).astype(np.float32)
    data_next_np = data_next.as_matrix().squeeze()
    net = RecurrentNet(architecture=[data_next_np.shape[0], 64, 32, 1])
    rmstd = RunningMeanStd(shape=data_next_np.shape)

    normalizer = lambda x: (x - rmstd.mean)/rmstd.std

    opt = Adam(net.parameters(), lr=1e-4)

    for epoch in range(100):
        generator = hdf_generator(path,chunk_size=100000,pair='ETHBTC', columns=['p', 'P', 'x', 'c', 'b', 'a', 'o', 'h', 'l', 'v', 'n'])()
        net.reset()
        try:
            while True:
                error = 0.
                abs_error = 0.
                for i in range(steps):
                    data_now = data_next
                    data_next = next(generator).astype(np.float32)
                    price = data_next.p
                    target_price = tt(np.asarray([data_next.p]))
                    data_now = data_now.as_matrix().flatten().reshape(1,1,-1)
                    
                    rmstd.update(data_now)
                    data_now = normalizer(data_now)
                    prediction = net(tt(data_now)) 
                    data_next = normalizer(data_next)
                    target_price = tt(np.asarray(data_next[0]))
                    error += (target_price-prediction)**2
                    abs_error += tor.sqrt(error)

                error = error/float(steps) * 0.5
                opt.zero_grad()
                error.backward(retain_graph=True)
                opt.step()
                print("{} epoch, error: {}, price_change: {}".format(epoch, abs_error.data.numpy(), price), end='\r')


                    #print(target_price, data_next.as_matrix(), data_now.as_matrix())
        except Exception as e:
            raise e
            break
                    
            




