#!/usr/bin/env python
import sys, os, time

import numpy as np
sys.path.insert(0, os.path.abspath("envs"))
from tetris import TetrisEnv

class Learner:
    def __init__(self):
        self.env = TetrisEnv()
        self.env.reset()

    def evaluate(self, state):
        # f1
        
        # f2
        
        # f3
        
        # f4
        
        # f5

    def test(self):
        pass

class Player:
    def __init__(self):
        # run a random policy on the tetris simulator
        # np.random.seed(1)
        self.env = TetrisEnv()
        self.env.reset()
        self.env.render()

    def play(self):
        for _ in range(50):
            actions = self.env.get_actions()
            print("Actions:")
            print(actions)
            rot = int(input("Rotation: "))
            self.env.render(rot=rot)
            mov = int(input("Movement: "))
            action = [rot,mov]
            # action = actions[int(input())]
            # action = actions[np.random.randint(len(actions))]
            state, reward, done, _ = self.env.step(action)
            print("Reward: {}".format(reward))
            print("Next state: ")
            print(state.field)
            if done:
                break
            self.env.render()

if __name__ == "__main__":
    agent = Player()
    agent.play()
    