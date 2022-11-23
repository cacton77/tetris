#!/usr/bin/self.env python
import sys, os, time
import json
import numpy as np
from operator import xor
from math import pi, cos
from functools import partial
import time
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from utils import features, action_independent_features
sys.path.insert(0, os.path.abspath("self.envs"))
from tetris import TetrisEnv

# pieces: O, I, L, J, T, S, Z
PIECES = [np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]),
np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]),
np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]),
np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]),
np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]),
np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]),
np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])]

class TetrisPlayer:
    def __init__(self, weights=None, verbose=False, animate=False):
        self.env = TetrisEnv(training=False)
        self.env.reset()

        if weights is not None:
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
            plt.style.use('classic')
            self.cmap = 'plasma'
            # fig = plt.figure()
            r, c = 2, 4
            fig, axs = plt.subplots(r,c)
            axs[0,0].tick_params(left=False, right=False, labelleft=False , labelbottom=False, bottom=False)
            for i in range(r):
                for j in range(c):
                    if i == 0 and j == 0: continue
                    axs[i,j].tick_params(left=False, right=False, labelleft=False , labelbottom=False, bottom=False)
                    axs[i,j].set_visible(False)
            axs[0,0].set_title('Next Piece')
            self.im_np = axs[0,0].imshow(PIECES[self.env.state.next_piece], vmin=0, vmax=1, cmap=self.cmap, interpolation='none')
            self.ax = fig.add_subplot(111)
            self.ax.set_title('TETRIS')
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

        ''' Game state rendering '''
        if self.verbose:
            os.system('cls' if os.name == 'nt' else 'clear')
            self.env.render()
            print(f'Score = {self.score}')
            print(f'Lines Cleared = {self.lines_cleared}')
        
        ''' Animation '''
        if self.animate:
            try:
                lowest = np.min(state.field[np.nonzero(state.field)])
            except: 
                lowest = 0
            self.im_np.set_data(PIECES[state.next_piece])
            self.im.set(clim=(0,state.turn - lowest + 10))
            self.im.set_data(np.flip(state.field - lowest + 10*(state.field!=0)))
            self.score_label.set_text(f'Score: {self.score}\nLines Cleared: {self.lines_cleared}')
            if self.done: 
                self.im.set_data(np.flip(state.field - lowest + 10*(state.field!=0)))
                self.ax.text(1, 6, 'GAME OVER', color='red', fontsize=20)
            return

    def end(self):
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


def run_experiment(filename, iters):
    of = open(filename)
    save_state = json.load(of)
    results = []
    print("Format: (avg lines, avg score, avg time)")
    for i in range(len(save_state) - 1):
        i = 91
        lines_cleared = []
        scores = []
        times = []
        for _ in range(iters):
            start = time.time()
            weights = np.array(save_state[str(i)]['mu'])
            agent = TetrisPlayer(verbose=False, animate=False, weights=weights)
            agent.play()
            lines_cleared.append(agent.lines_cleared)
            scores.append(agent.score)
            times.append(time.time() - start)

        results.append((np.mean(lines_cleared), np.mean(scores), np.mean(times)))
        print(f"Results for iteration {i}: {results[-1]}")
        np.save('results', results)


if __name__ == "__main__":
    # run_experiment('cem_save_state.json', 20)
    agent = TetrisPlayer(verbose=False,animate=True)
    agent.play()
    

