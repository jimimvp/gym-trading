from gym.envs.registration import register

from .binance_trading import *


capital = {
    
    'ETH': 7.2,
    'BTC': 0.4

}

register(
    id='BinanceTradingETHBTC-v1',
    kwargs = {'pair' : 'ETHBTC', 'capital' : capital,
    'fee':0.0015, 'percentage_tolerance':0.25},
    entry_point='gym_trading.envs:BinanceTradingEnv',
)

