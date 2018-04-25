import gym
from gym import spaces
import numpy as np

def CryptoTradingEnv(gym.Env):


    def __init__(self, num_currencies=100, freq=1, capital=100, fee=0.15, percentage_tolerance=0.25):
        super(CryptoTradingEnv, self).__init__()
        self.high = np.inf
        self.low = -np.inf
        self.action_space = spaces.Box(high=self.high, low=self.low, shape=(num_currencies,))

        curr_info_vars = ['24hourvolume' '1hourvolume', 'lastminclose', 'lastminopen', 'lastminvol', 'mcap', 'price', 'invested - curr_worth']
        self.observation_space = spaces.Box(high=self.high, low=self.low, shape=(num_currencies*len(curr_info_vars),))
        self.percentage_tolerance = percentage_tolerance
        self.unfilled_orders = []

        self.bought = np.zeros(num_currencies)
        self.feefactor = fee/100.

    def step(self,action):
        """
            Eeach element corresponds to a cryptocurrency amount -amount for selling and amount for buying
            if -amount is greater than amount owned, everything is sold, if amount buy is
            greater that capital available, all capital is used + fee. Maybe also incorporate a possibility
            for price as a percentage of the curr price? (Softmax + 0.5)/2. so that 25% of current price or 125% of
            current price can be used. 
        """
        action = action.reshape(-1, 2)
        assert np.all(np.isclose(action[:, 1], 1., atol=self.percentage_tolerance)), 'Price is too high or too low'

    def get_curr_prices(self):
        return np.zeros(self.observation_space.shape[0]/2)

    def try_to_fill_orders(self):
        pass

    def cancel_order(self):
        pass

    def buy(self, crypto, price):
        pass


    
    def sell(self, crypto):
        pass



    def
