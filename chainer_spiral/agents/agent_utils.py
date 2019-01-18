import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import cv2
import numpy as np

def preprocess_image(x):
    """ function to preprocess image from observation """
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)  # unit8, 2d matrix
    x = x.astype(np.float32) / 255.0
    x = np.reshape(x, [1, 1] + list(x.shape))
    return x

def preprocess_obs(obs, imsize):
    """ function to preprocess observation from env """
    c = obs['image']
    x = obs['position']
    q = obs['prob']

    # image
    c = preprocess_image(c)

    # position
    x /= float(imsize * imsize)
    x = np.asarray(x, dtype=np.float32) 
    x = np.reshape(x, (1, 1))

    # prob
    q = np.asarray(q, dtype=np.float32)
    q = np.reshape(q, (1, 1))
    
    # return state as a tuple
    return c, x, q

def pack_action(act):
    a1, a2 = act  # sampled actions by policy net
    return {'position': int(a1.data),
            'pressure': 1.0,
            'color': (0, 0, 0),
            'prob': int(a2.data)}

def compute_auxiliary_reward(past_reward, 
                            past_act, 
                            n_episode,
                            max_episode_steps,
                            staying_penalty,
                            empty_drawing_penalty):
    empty = True
    drawing_steps = 0

    last_a1, last_a2 = None, None

    for t in range(max_episode_steps):
        a1 = past_act[n_episode, t]['position']
        a2 = past_act[n_episode, t]['prob']
        
        if empty and a2: 
            empty = False

        if t > 0:
            if last_a2 and a2:
                drawing_steps += 1
            if not a2:
                drawing_steps = 0

            if last_a1 == a1:
                past_reward[n_episode, t] -= staying_penalty

        last_a1 = a1
        last_a2 = a2

    if empty:
        past_reward[n_episode, max_episode_steps - 1] -= empty_drawing_penalty
    
    return past_reward

class ObservationSaver(object):
    def __init__(self, outdir, rollout_n, imsize):
        self.outdir = outdir
        self.rollout_n = rollout_n
        self.imsize = imsize

        # create directory to save png files
        self.target_dir = os.path.join(self.outdir, 'final_obs')
        if not os.path.exists(self.target_dir):
            os.mkdir(self.target_dir)

        # init figure
        self.fig = plt.figure(figsize=(7, 7))
        gs = gridspec.GridSpec(self.rollout_n, 2)
        self.ims_real, self.ims_fake = [], []
        for n in range(self.rollout_n):
            ax = plt.subplot(gs[n, 0])
            self.ims_fake.append(
                ax.imshow(np.zeros((self.imsize, self.imsize)), vmin=0, vmax=1, cmap='gray')
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if n == 0:
                ax.set_title('Fake data')

            ax = plt.subplot(gs[n, 1])
            self.ims_real.append(
                ax.imshow(np.zeros((self.imsize, self.imsize)), vmin=0, vmax=1, cmap='gray')
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if n == 0:
                ax.set_title('Real data')

    def save(self, fake_data, real_data, update_n):
        for n in range(self.rollout_n):
            self.ims_fake[n].set_data(fake_data[n][0, 0])
            self.ims_real[n].set_data(real_data[n].data[0, 0])
        self.fig.suptitle(f"Update = {update_n}")
        savename = os.path.join(self.target_dir, f"obs_update_{update_n}.png")
        plt.savefig(savename)


