from environments import MyPaintEnv
import numpy as np
from gym import wrappers

class RandomAgent(object):
    """ simplest agent """
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, observation, reward, done):
        a = self.action_space.sample()
        a['color'] = (0, 0, 0)
        print(f"taking action {a}")
        return a

if __name__ == '__main__':
    env = MyPaintEnv()
    agent = RandomAgent(env.action_space)

    r = 0
    done = False,
    o = env.reset()
    for t in range(3):
        a = agent.act(o, r, done)
        o, r, done, _ = env.step(a)
    
    import matplotlib
    matplotlib.use('GTK3Agg')
    import matplotlib.pyplot as plt
    plt.imshow(o['image'])
    plt.show()

    env.close()