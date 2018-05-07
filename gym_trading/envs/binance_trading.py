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
import os

random.seed(123)



path = os.environ['BINANCE_DATA_PATH']

def hdf_generator(path, chunk_size=10000, pair=None, columns=None):
    """
        Returns a new generator object.
    """
    def generator():
        iterator = pd.read_hdf(path, chunksize=chunk_size)
        for chunk in iterator:
            if pair:
                chunk = chunk[chunk.s == pair]
            if columns:
                chunk = chunk[columns]
            for i, row in chunk.iterrows():
                yield row
        return None
    return generator

class BinanceTradingEnv(gym.Env):



    def __init__(self, pair='ETHBTC', freq=1, capital=None, fee=0.0015, percentage_tolerance=0.25):
        super(BinanceTradingEnv, self).__init__()
        self.high = 1
        self.low = -1
        self.pair = pair
        self.curr_info_vars = ['p', 'P', 'x', 'c', 'b', 'a', 'o', 'h', 'l', 'v', 'n']
        self.f = freq

        #Base and top currency
        self.top = pair[:3]
        self.base = pair[3:]
        self.generator = None
        #How much percentage from capital to sell maximum per order
        self.pt = percentage_tolerance
        self.curr_chunk = None
        self.df_idx = 0
        self.chunk_idx = 0

        if capital is None:
            #BTC ETH 
            self.capital = Capital({
                "ETH": 6.,
                "BTC" : 0.2
            })
        else:
            self.capital = Capital(capital)

        self.observation_space = spaces.Box(high=np.inf, low=-np.inf, shape=(len(self.curr_info_vars)+4,), dtype=np.float32)
        #Made only for one currency pair
        self.action_space = spaces.Box(high=1, low=-1, shape=(1,), dtype=np.float32)

    @property
    def ticker(self):
        df = next(self.generator)
        return df.astype(np.float32)
        

    def reset(self):
        #Reset the generator to the beginning
        self.generator = hdf_generator(path=path, pair=self.pair, columns=self.curr_info_vars)()
        return np.hstack([self.ticker.as_matrix().squeeze(), 
            self.capital.win[0],  self.capital.balances[0]]).astype(np.float32).squeeze()

    def process_action(self, a):
        """
            Returns orders to be made
        """
        sell_symbol = self.top if a < 0 else self.base
        order_type = OrderType.SELL_ORDER if a < 0 else OrderType.BUY_ORDER
        amount = self.capital[sell_symbol]*np.abs(a) * self.pt
        #Pay the ask price for now...
        order = Order(pair=self.pair, type=order_type, amount=amount, price=self.ticker.a, i=12353513)    

        #capital.process_order(order)
        # Order maybe won't go through
        if np.random.choice([True,False], p=[0.95, 0.05]):
            self.capital.process_order(order)
            return True
        else:
            return False


    def step(self,action):
        """
            Eeach element corresponds to a cryptocurrency amount -amount for selling and amount for buying
            if -amount is greater than amount owned, everything is sold, if amount buy is
            greater that capital available, all capital is used + fee. Maybe also incorporate a possibility
            for price as a percentage of the curr price? (Softmax + 0.5)/2. so that 25% of current price or 125% of
            current price can be used. 
        """
        went_through = self.process_action(action)
        delta, keys = self.capital.win
        balance, _ = self.capital.balances
        
        #'Calculate everything in btc'
        reward = 0.
        for i, k in enumerate(keys):
            if k != 'BTC':
                reward+=delta[i]
            reward+=delta[i]*self.ticker.a

        obs = self.ticker.as_matrix().squeeze()
        #Give delta between current state and beginning
        #print(obs.shape, delta.shape, balance.shape)
        #input()
        obs = np.hstack([obs, delta, balance]).astype(np.float32).squeeze()
        return obs, reward, False, {}

import time
if __name__ == '__main__':

    env = BinanceTradingEnv()
    obs = env.reset()
    for i in range(100):
        obs, r, d, i = env.step(env.action_space.sample())
        print(r)



