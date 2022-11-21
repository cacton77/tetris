#!/usr/bin/self.env python
import sys, os, time
import json
import numpy as np
from operator import xor
from math import pi, cos
import time
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
            fig = plt.figure()
            self.im = plt.imshow(self.env.state.field, cmap='viridis', interpolation='none')
            anim = FuncAnimation(fig, self.step, frames=10000, interval=1000)
            plt.show()

    def step(self, t=0):
        state = self.env.state.copy()
        actions = self.env.get_actions()
        Q = len(actions)*[0]
        fs = np.zeros((len(actions),8))
        # print(actions)
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
            self.im.set_data(np.flip(state.field))
            return [self.im]
            # return self.ax
            # plt.imshow(np.flip(state.field, 0), cmap='viridis')
            plt.show()       

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


def run_experiment(filename, iters):
    of = open(filename)
    save_state = json.load(of)
    results = []
    print("Format: (avg lines, avg score, avg time)")
    for i in range(len(save_state) - 1):
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
    # agent = TetrisPlayer(verbose=False,animate=False)
    # agent.play()
    run_experiment('./cem_save_state.json', 10)
    

