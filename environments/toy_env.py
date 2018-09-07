import gym
import numpy as np

class ToyEnv(gym.Env):
    action_space = None
    observation_space = None
    reward_range = None
    viewer = None

    metadata = {
        'render.modes': ['rgb_array'],
    }

    def __init__(self, imsize):
        """ init env """
        super().__init__()
        
        self.imsize = imsize

        # action_space
        self.action_space = gym.spaces.Dict({
            'position': gym.spaces.Discrete(self.imsize ** 2),
            'pressure': gym.spaces.Box(low=0, high=1.0, shape=()),
            'color': gym.spaces.Box(low=0, high=1.0, shape=(3,)),
            'prob': gym.spaces.Discrete(2)
        })        

        # observation space
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(self.imsize, self.imsize, 3), dtype=np.uint8),
            'position': gym.spaces.Discrete(self.imsize ** 2),
            'pressure': gym.spaces.Box(low=0, high=1.0, shape=()),
            'color': gym.spaces.Box(low=0, high=1.0, shape=(3,)),
            'prob': gym.spaces.Discrete(1)
        })

        self.reset()

    def reset(self):
        self.image = np.ones((self.imsize, self.imsize, 3)) * 255.0
        self.image = self.image.astype(np.uint8)
        o = {'image': self.image,
            'position': 0,
            'pressure': 0,
            'color': (0, 0, 0),
            'prob': 0}
        # return observation, reward, done, and info
        return o

    def step(self, action):
        x = action['position']
        p = action['pressure']
        c = action['color']
        q = action['prob']

        p1, p2 = self.convert_x(x)

        if q:
            self.image[p1, p2, :] = 0
        
        o = {'image': self.image,
            'position': x,
            'pressure': p,
            'color': c,
            'prob': q}
        
        # return observation, reward, done, and info
        return o, 0, False, None

    def convert_x(self, x):
        """ convert position id -> a point (p1, p2) """
        assert x < self.imsize ** 2
        p1 = x % self.imsize
        p2 = x // self.imsize
        return int(p1), int(p2)

    def render(self, mode='rgb_array'):
        """ render the current drawn picture image for human """
        if mode == 'rgb_array':
            return self.image
        else:
            raise NotImplementedError

    def close(self):
        pass
    
    def seed(self, seed=None):
        pass


if __name__ == '__main__':
    env = ToyEnv(3)
    steps = 4
    for t in range(steps):
        a = env.action_space.sample()
        print(a)
        o, _, _, _ = env.step(a)
    
    import matplotlib.pyplot as plt
    plt.imshow(o['image'])
    plt.show()
