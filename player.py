#!/usr/bin/self.env python
import sys, os, time
import json
import numpy as np
from operator import xor
from math import pi, cos
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from utils import features, action_independent_features
sys.path.insert(0, os.path.abspath("self.envs"))
from tetris import TetrisEnv


class Player:
    def __init__(self):
        self.env = TetrisEnv(training=False)
        self.env.reset()

    def play(self):
        save_state = {}

        if os.path.exists('./cem_save_state.json'):
            with open('./cem_save_state.json') as of:
                save_state = json.load(of)
        else:
            sys.exit("No optimization history.")

        last_iters = save_state['last_iters']

        theta = np.mean(save_state[str(last_iters)]['mu'])

        self.tetris_fun(self, theta, training=False, verbose=True)
        

    def tetris_fun(self, theta, samples=20, training=True, verbose=False):
        self.env.reset()
        self.env.render()
        fxs = samples*[0]
        for s in range(samples):
            fx = 0
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                state = self.env.state.copy()
                actions = self.env.get_actions()
                Q = len(actions)*[0]
                fs = np.zeros((len(actions),8))
                # print(actions)
                for i in range(len(actions)):
                    a_i = actions[i]
                    next_state, reward, done, _ = self.env.step(a_i)
                    fs[i,:] = features(state, reward, next_state, verbose=False)
                    Q[i] = theta.dot(fs[i,:])
                    self.env.set_state(state)
                V_min = np.min(Q)
                i_min = np.argmin(Q)
                a_min = actions[i_min]

                state, reward, done, _ = self.env.step(a_min)
                self.env.render()
                fx += reward
                print(f'Score = {fx}')
            
                if done:
                    if verbose: 
                        print("+==========================================+")
                        print("+                  TETRIS                  +")
                        print("+==========================================+\n")
                        print(f'Weights: {theta}')
                        self.env.render()
                        print(f'\nLines cleared: {state.cleared}')
                        print(f'Score: {fx}\n')
                    self.env.reset()
                    fxs[s] = fx
                    break
        fx_avg = np.mean(fxs)
        if verbose: 
            print(f'High score: {np.max(fxs)}')
            print(f'Average Score: {fx_avg}')
        return fx_avg

if __name__ == "__main__":
    agent = Player()
    weights = np.array([0.36335872, -0.39474949, 0.36761178, 0.52811635, 0.10330465, 0.11987217, 0.21252479, 0.32646567])
    agent.tetris_fun(weights, verbose=True)
    
