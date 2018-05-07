import gym
from gym import spaces
import numpy as np
import pandas as pd
from collections import namedtuple
import copy
from enum import Enum

class OrderType(Enum):

    SELL_ORDER = 'SELL'
    BUY_ORDER = 'BUY'


Order = namedtuple('Order', ['amount', 'type', 'pair', 'i', 'price'])


class Capital(object):


    def __init__(self, capital_dict, fee=0.0015):
        self.capital_dict = capital_dict
        self.start_capital = copy.deepcopy(self.capital_dict)
        self.start_cap_values = np.asarray(list(self.start_capital.values()))
        self.start_cap_keys = self.start_capital.keys()
        self.fee = fee

    def process_orders(self, orders):
        for order in orders:
            self.process_order(order)
            

    def process_order(self, order):
        assert isinstance(order.type, OrderType), 'Unknown order type'
            
        top, base = order.pair[:3], order.pair[3:]
        a, p = order.amount, order.price

        if order.type == OrderType.SELL_ORDER:
            #Buting base currency wit top currency
            selling = top
            buying = base
        elif order.type == OrderType.BUY_ORDER:
            #Reverse
            selling = base
            buying = top

        assert a < self.capital_dict[selling], 'Selling more than available'

        self.capital_dict[selling] -= float(a)
        self.capital_dict[buying] += float((1.-self.fee)*a*p)

    def __getitem__(self, symbol):
        return self.capital_dict[symbol]

    @property
    def win(self):
        delta = np.asarray(list(self.capital_dict.values())) - self.start_cap_values
        keys = self.capital_dict.keys()
        return delta, keys

    @property
    def balances(self):
        return np.asarray(list(self.capital_dict.values())), self.capital_dict.keys()

import numpy as np

if __name__ == '__main__':
    cap_dict = {
                    'ETH' : 6.,
                    'BTC' : 0.2
                }
    
    cap = Capital(cap_dict)

    #Generate ETHBTC orders
    orders = [
        Order(amount=0.2, price=0.1, pair='ETHBTC', i=1125326246, type=OrderType.SELL_ORDER)
    ] 

    cap.process_orders(orders)
    cd = cap.capital_dict

    assert np.isclose(cd['ETH'], (6. - 0.2), atol=1e-8), 'Order not processed correctly, ETH {} in stead of 5.8'.format(cd['ETH'])
    #Because of fee
    assert np.isclose(cd['BTC'], (.2 + 0.1*0.2*0.9985), atol=1e-8), 'Order not processed correctly, ETH {} in stead of 5.8'.format(cd['ETH'])

    assert cap.win[0].shape[0]==2
    assert cap.win[0].shape[0]==2

    print('Tests completed successfully')
