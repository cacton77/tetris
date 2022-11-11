#!/usr/bin/env python3

import sys, os

import numpy as np
sys.path.insert(0, os.path.abspath("envs"))
from tetris import TetrisEnv

class Learner:
    def __init__(self):
        self.env = TetrisEnv()
        self.env.reset()

    def evaluate(self, weight, feature):
        '''
        Evaluation function V is a combination of these feature functions using generally a weighted
        linear sum.
        '''

    def feature_5(self):
        '''
        f5: Number of holes: the number of empty cells with at least one filled cell above.
        '''
        holes = 0

        for r in range(self.env.n_rows-1, -1, -1):
            if r == 20:
                pass

            else:
                for c in range(self.env.n_cols): 
                    if (self.env.state.field[r, c] == 0) and (self.env.state.field[r+1, c] > 0):
                        holes += 1

        return holes

    def feature_6(self):
        '''
        Cumulative wells: the sum of the accumulated depths of the wells.
        '''
        depths = 0

        r_list = []

        for c in range(self.env.n_cols):
            if c == 0:
                pass

            elif c == 9:
                pass

            else:
                for r in range(self.env.n_rows-2, -1, -1):
                    if (self.env.state.field[r, c] > 0):
                        r_list.append(r)
                        r_surface = max(r_list)

                        for c in range(1, self.env.n_cols-2, 1):
                            if (self.env.state.field[r_surface-1, c] > 0):


        return depths

class Player:
    def __init__(self):
        self.env = TetrisEnv()
        self.env.reset()
        self.env.render()

    def play(self):
        # Manual policy
        # actions = self.env.get_actions()
        # for _ in range(50):
        #     actions = self.env.get_actions()
        #     print("Actions:")
        #     print(actions)
        #     rot = int(input("Rotation: "))
        #     self.env.render(rot=rot)
        #     mov = int(input("Movement: "))
        #     action = [rot, mov]
        #     state, reward, done, _ = self.env.step(action)
        #     print("Reward: {}".format(reward))
        #     print("Next state: ")
        #     print(state.field)
        #     if done:
        #         print("Game Over!")
        #         break
        #     self.env.render()

        # Random policy
        for _ in range(50):
            actions = self.env.get_actions()
            action = actions[np.random.randint(len(actions))]
            state, reward, done, _ = self.env.step(action)

            holes = Learner.feature_5(self)
            depths = Learner.feature_6(self)

            print('#####################################################################')
            print('Number of holes: {}'.format(holes))
            print('Sum of accumulated depths: {}'.format(depths))
            print("Next state: ")
            print(state.field)
        
            if done:
                break
            self.env.render()

if __name__ == "__main__":
    agent = Player()
    agent.play()

