#!/usr/bin/env python
import sys, os, time
import numpy as np
from operator import xor
from math import pi, cos
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.path.insert(0, os.path.abspath("envs"))
from tetris import TetrisEnv

class Learner:
    def __init__(self):
        self.env = TetrisEnv()
        self.env.reset()

    def learn(self):
        n = 2 # . . . . . . . . Number of parameters

        bounds = n*[(-5.12, 5.12)] # . . . . . . . Bounds for parameters
        x, fx_f, iters = self.nelder_mead(self.rastrigin_fun, n=n, bounds=bounds, verbose=True)
        print("Rastrigin : x = {}, iters = {}".format(x, iters))


        bounds = n*[(-5.12, 5.12)] # . . . . . . . Bounds for parameters
        x, fx_f, iters = self.nelder_mead(self.sphere_fun, n=n, bounds=bounds, verbose=True)
        print("Sphere iters : x = {}, iters = {}".format(x, iters))

        bounds = n*[(-5.12, 5.12)] # . . . . . . . Bounds for parameters
        x, fx_f, iters = self.nelder_mead(self.rosenbrock_fun, n=n, bounds=bounds, verbose=True)
        print("Rosenbrock iters : x = {}, iters = {}".format(x, iters))

    # __________________________________________________________
    # FUNCTIONS FOR TESTING OPTIMIZATION
    # - Rastrigin function
    # - Sphere function
    # __________________________________________________________

    def rastrigin_fun(self, x):
        A = 10
        n = len(x)
        fx = A*n
        for i in range(n):
            fx += x[i]**2 - A*np.cos(2*pi*x[i])
        return fx

    def sphere_fun(self, x):
        n = len(x)
        fx = 0
        for i in range(n):
            fx += x[i]**2
        return fx

    def rosenbrock_fun(self, x):
        n = len(x)
        fx = 0
        for i in range(n-1):
            fx += 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return fx

    # __________________________________________________________
    # OPTIMIZATION FUNCTIONS
    # - Nelder-Mead (Simplex Search)
    # __________________________________________________________

    def nelder_mead(self, f, n, bounds, tol=1e-12, maximize=False, verbose=False):
        # Initialize simplex
        x = np.zeros((n+1, n))
        for i in range(n):
            b_l, b_u = bounds[i][0], bounds[i][1]
            b_r, b_c = b_u - b_l, (b_l + b_u)/2 # range and center of bounds
            x[:,i] = b_r*(np.random.rand(n+1,)-0.5) + b_c

        iters = 0

        while True:
            # 1. Evaluate vertices of simplex
            fx = np.zeros((n+1,))
            for i in range(n+1):
                fx[i] = f(x[i,:])

            std_dev = np.std(fx)

            if std_dev < tol:
                x_f = np.mean(x, axis=0)
                fx_f = f(x_f)
                if verbose:
                    if n == 2:
                        p = 1000
                        x_1 = np.linspace(bounds[0][0], bounds[0][1], p)
                        x_2 = np.linspace(bounds[1][0], bounds[1][1], p)
                        x_1, x_2 = np.meshgrid(x_1, x_2)
                        fx12 = f([x_1,x_2])

                        fig = plt.figure()
                        ax = plt.axes(projection='3d')

                        surf = ax.plot_surface(x_1, x_2, fx12, cmap='viridis')
                        ax.scatter(x_f[0], x_f[1], fx_f, s=10, c='#eb000f', marker='o')
                        # ax.set_title('Rastrigin Function')
                        plt.show()
                return x_f, fx_f, iters

            # Begin iteration
            iters += 1

            # Order according to values at vertices
            order = np.argsort(fx) # indices sorted lowest to highest

            # Find worst vertex
            if maximize:
                order = np.flip(order)
            h = order[n]
            x_h = x[h,:]
            fx_h = f(x_h)

            # 2. Calculate centroid of other vertices
            x_sum = np.sum(x, axis=0)
            x_sum -= x_h
            x_o = x_sum/n

            # 3. Reflection: compute and evaluate reflected point x_r
            alpha = 1. # weight alpha > 0
            x_r = x_o + alpha*(x_o - x_h)
            fx_r = f(x_r)

            # If the reflected point is better than the second worst x_n,
            # but not better than the best x_1, then obtain a new
            # simplex by replacing the worst point x_h with the reflected
            # point x_r and go to step 1

            x_1, x_n = x[order[0],:], x[order[n-1],:]
            fx_1, fx_n = f(x_1), f(x_n)

            if maximize:
                if fx_r <= fx_1 and fx_r > fx_n:
                    x[h,:] = x_r
                    continue
            else:
                if fx_r >= fx_1 and fx_r < fx_n:
                    x[h,:] = x_r
                    continue

            # 4. Expansion: if the reflected point is best so far, compute
            # the expanded point
            if xor(fx_r < fx_1, maximize):
                gamma = 2. # weight gamma > 1
                x_e = x_o + gamma*(x_r - x_o)
                fx_e = f(x_e)
                
                # If the expanded point is better than the reflected point,
                # then obtain a new simplex by replacing the worst x_h with
                # the expanded point x_e and go to step 1
                if xor(fx_e < fx_r, maximize):
                    x[h,:] = x_e
                    continue
                else:
                    x[h,:] = x_r
                    continue

            # 5. Contraction: x_r is worse than x_n. If it is the worst, 
            # compute the contracted point outside the simplex
            rho = 0.5 # weight 0 < rho <= 0.5
            if xor(fx_r < fx_h, maximize):
                x_c = x_o + rho*(x_r - x_o)
                fx_c = f(x_c)

                # If the contracted point is better than the reflected point, 
                # contract the simplex by moving x_h to x_c
                if xor(fx_c < fx_r, maximize):
                    x[h,:] = x_c
                    continue

            # Otherwise compute the contracted point on the inside
            else:
                x_c = x_o + rho*(x_h - x_o)
                fx_c = f(x_c)
                
                # If the contracted point is better than the worst point,
                # then obtain a new simplex by replacing the x_h with 
                # the contracted point x_c and go to step 1
                if xor(fx_c < fx_h, maximize):
                    x[h,:] = x_c
                    continue

            # 6. Shrink: Replace all points except the best, x_1
            sigma = 0.5
            for i in range(n+1):
                if i == order[0]: continue
                x[i,:] = x_1 + sigma*(x[i,:] - x_1)
                
                
                
    def evaluate(self, weights, features):
        '''
        Evaluation function V is a combination of these feature functions using generally a weighted
        linear sum.
        '''
        pass

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

    def action_independent_features(self):
        field = self.env.state.field.copy()

        # Compute Preliminaries
        #   f3-f5: Binarize field and add 'borders' for computing 
        field[field > 0] = 1
        top_wall = np.ones(field.shape[1])
        left_right_walls = np.ones((field.shape[0] + 1, 1))
        bordered_field = np.hstack([left_right_walls, np.vstack([top_wall, field]), left_right_walls])

        #   Compute the change in each row and column when the row below is subtracted
        delta_row = (bordered_field[:-1] - bordered_field[1:])
        delta_column = (bordered_field[:, :-1] - bordered_field[:,  1:])

        #   f6 - f8: Determine wells and holes 
        r, c = np.nonzero(field)
        heights = np.array([100] + [np.max(r, where=(c == i), initial=-1) for i in range(10)] + [100])
        _, thresholds = find_peaks(-heights, threshold=0.5)
        wells = np.minimum(thresholds['left_thresholds'], thresholds['right_thresholds'])
        holes = np.argwhere(delta_row == -1)

        # Compute features
        f3 = (delta_column != 0).sum()
        f4 = (delta_row != 0).sum()
        f5 = (delta_row == -1).sum()
        f6 = int((wells * (wells + 1) // 2).sum())
        f7 = int(np.sum([heights[hole[1]] - hole[0] + 1 for hole in holes]))
        f8 = len(np.unique(holes[:, 0]))

        return [f3, f4, f5, f6, f7, f8]



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
            a_indep = Learner.action_independent_features(self)

            print('#####################################################################')
            print('Number of holes: {}'.format(holes))
            print('Sum of accumulated depths: {}'.format(depths))
            print(f'Action-independent features: {a_indep}')
            print("Next state: ")
            print(state.field)
            input()
        
            if done:
                break
            self.env.render()

if __name__ == "__main__":
    agent = Player()
    agent.play()
    # learner = Learner()
    # learner.learn()
    
