import gym
import gym_trading



if __name__ == '__main__':

    env = gym.make('BinanceTradingETHBTC-v1')
    #Reset env

    obs = env.reset()
    assert obs.shape == env.observation_space.shape, '{} vs {}'.format(obs.shape, env.observation_space.shape)
    for i in range(100):
        a = env.action_space.sample()
        o, r, d, i = env.step(a)
        assert o.shape == obs.shape

