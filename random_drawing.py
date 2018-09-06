from environments import MyPaintEnv
import numpy as np
from gym import wrappers
import argparse

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_resolution', type=int, default=64)
    parser.add_argument('--pos_resolution', type=int, default=32)
    parser.add_argument('--max_episode_steps', type=int, default=10)
    args = parser.parse_args()

    env = MyPaintEnv(imsize=args.image_resolution, 
                        pos_resolution=args.pos_resolution, 
                        max_episode_steps=args.max_episode_steps)
    
    # Gym's monitor does not support small image inputs
    if args.image_resolution >= 30:
        env = wrappers.Monitor(env, directory='./tmp', force=True)
    
    agent = RandomAgent(env.action_space)

    reward = 0
    done = False
    ob = env.reset()

    for t in range(args.max_episode_steps):
        action = agent.act(ob, reward, done)
        if t == 0:
            action['prob'] = 0
        ob, reward, done, _ = env.step(action)

    env.close()

    if args.image_resolution < 30:
        # save the final observation instead of monitor
        import matplotlib
        matplotlib.use('Cairo')
        import matplotlib.pyplot as plt
        plt.imshow(ob['image'])
        plt.savefig('./tmp/random.png')

