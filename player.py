#!/usr/bin/self.env python
import sys, os, time
import json
import numpy as np
from operator import xor
from math import pi, cos
from functools import partial
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from utils import features, action_independent_features
sys.path.insert(0, os.path.abspath("self.envs"))
from tetris import TetrisEnv


class TetrisPlayer:
    def __init__(self, weights=None, verbose=False, animate=False):
        self.env = TetrisEnv(training=False)
        self.env.reset()

        if weights:
            self.weights = weights
        else:
            save_state = {}

            if os.path.exists('./cem_save_state.json'):
                with open('./cem_save_state.json') as of:
                    save_state = json.load(of)
            else:
                sys.exit("No optimization history.")

            last_iters = save_state['last_iters']

            self.weights = np.array(save_state[str(last_iters)]['mu'])

        self.done = False
        self.lines_cleared = 0
        self.score = 0

        self.verbose = verbose
        if self.verbose:
            self.env.render()

        self.animate = animate
        if self.animate:
            self.cmap = 'plasma'
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.tick_params(left=False, right=False, labelleft=False , labelbottom=False, bottom=False)
            self.im = plt.imshow(self.env.state.field, vmin=0, vmax=0, cmap=self.cmap, interpolation='none')
            self.score_label = plt.xlabel('Score: 0\nLines Cleared: 0')
            anim = FuncAnimation(fig, self.step, frames=10000, interval=5)
            plt.show()

    def step(self, t=0):
        state = self.env.state.copy()
        actions = self.env.get_actions()
        Q = len(actions)*[0]
        fs = np.zeros((len(actions),8))
        for i in range(len(actions)):
            a_i = actions[i]
            next_state, reward, done, _ = self.env.step(a_i)
            fs[i,:] = features(state, reward, next_state, verbose=False)
            Q[i] = self.weights.dot(fs[i,:])
            self.env.set_state(state)
        V_min = np.min(Q)
        i_min = np.argmin(Q)
        a_min = actions[i_min]

        state, reward, self.done, _ = self.env.step(a_min)
        self.lines_cleared = state.cleared
        self.score += reward

        if self.verbose:
            os.system('cls' if os.name == 'nt' else 'clear')
            self.env.render()
            print(f'Score = {self.score}')
            print(f'Lines Cleared = {self.lines_cleared}')
        if self.animate:
            if done: return 
            # im = plt.imshow(np.flip(state.field)-np.max(state.field), cmap=self.cmap, interpolation='none')
            lowest = np.min(state.field[np.nonzero(state.field)])
            self.im.set(clim=(0,state.turn - lowest + 10))
            self.im.set_data(np.flip(state.field - lowest + 10*(state.field!=0)))
            self.score_label.set_text(f'Score: {self.score}\nLines Cleared: {self.lines_cleared}')
            return
            # return self.im
            # return self.im
            # return [self.im]

    def end(self):
        if self.verbose: 
            print("+==========================================+")
            print("+                  TETRIS                  +")
            print("+==========================================+\n")
            print(f'Weights: {self.weights}')
            self.env.render()
            print(f'\nLines cleared: {self.lines_cleared}')
            print(f'Score: {self.score}\n')
        self.env.reset()

    def play(self):
        self.env.reset()
        while not self.done:
            self.step()
        self.end()
        
if __name__ == "__main__":
    agent = TetrisPlayer(verbose=False,animate=True)
    # agent.play()
    
