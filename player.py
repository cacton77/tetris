#!/usr/bin/env python
import sys, os, time
import numpy as np
from operator import xor
from math import pi, cos
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from utils import features, action_independent_features
sys.path.insert(0, os.path.abspath("envs"))
from tetris import TetrisEnv


class Player:
    def __init__(self):
        self.env = TetrisEnv()
        self.env.reset()
        self.env.render()

    def play(self):
        # Manual policy
        actions = self.env.get_actions()
        for _ in range(50):
            a_indep = action_independent_features(self.env.state.field.copy())
            print(f'Action-independent features: {a_indep}')
            actions = self.env.get_actions()
            print("Actions:")
            print(actions)
            rot = int(input("Rotation: "))
            self.env.render(rot=rot)
            mov = int(input("Movement: "))
            action = [rot, mov]
            state, reward, done, _ = self.env.step(action)
            print("Reward: {}".format(reward))
            print("Next state: ")
            print(state.field)
            if done:
                print("Game Over!")
                break
            self.env.render()

        # Random policy
        for _ in range(50):
            actions = self.env.get_actions()
            action = actions[np.random.randint(len(actions))]
            state, reward, done, _ = self.env.step(action)

            a_indep = action_independent_features(self.env.state.field.copy())

            print('#####################################################################')
            print(f'Action-independent features: {a_indep}')
            print("Next state: ")
            print(state.field)
            input()
        
            if done:
                break
            self.env.render()

if __name__ == "__main__":
    # agent = Player()
    # agent.play()
    learner = Learner()
    learner.learn()
    
