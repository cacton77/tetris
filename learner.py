#!/usr/bin/env python
import sys, os, time
import json
import random
import numpy as np
from operator import xor
from math import pi, cos
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from utils import features, action_independent_features
sys.path.insert(0, os.path.abspath("envs"))
from tetris import TetrisEnv

NO_NOISE = 0
CONSTANT_NOISE = 1
DECREASING_NOISE = 2

class Learner:
    def __init__(self):
        self.env = TetrisEnv(training=True)
        self.env.reset()

    def learn(self):
        n = 8 # . . . . . . . . Number of parameters

        b1 = (0., 1.)
        b2 = (-1., -0.75)
        b3 = (0., 1.)
        b4 = (0., 1.)
        b5 = (0., 1.)
        b6 = (0., 1.)
        b7 = (0., 1.)
        b8 = (0., 1.)
        bounds = [b1, b2, b3, b4, b5, b6, b7, b8]
        x, fx_f, iters = self.cross_entropy_method(self.tetris_fun, 
                                                   n=n, 
                                                   bounds=bounds, 
                                                   samples=25, 
                                                   function_samples=25,
                                                   N_top=4, 
                                                   noise=DECREASING_NOISE, 
                                                   maximize=True, normalize=True, verbose=True, checkpoints=True, load_save_state=True)
        print(x)
        return
        x, fx_f, iters = self.nelder_mead(self.tetris_fun, n=n, bounds=bounds, maximize=True, normalize=True, verbose=True)
        print("Tetris: x = {}, fx = {}, iters = {}".format(x, fx_f, iters))

        return

        bounds = n*[(-512, 512)] # . . . . . . . Bounds for parameters
        x, fx_f, iters = self.nelder_mead(self.rastrigin_fun, n=n, bounds=bounds, maximize=True, verbose=True)
        print("Rastrigin : x = {}, iters = {}".format(x, iters))


        bounds = n*[(-5.12, 5.12)] # . . . . . . . Bounds for parameters
        x, fx_f, iters = self.nelder_mead(self.sphere_fun, n=n, bounds=bounds, maximize=True, verbose=True)
        print("Sphere iters : x = {}, iters = {}".format(x, iters))

        bounds = n*[(-5.12, 5.12)] # . . . . . . . Bounds for parameters
        x, fx_f, iters = self.nelder_mead(self.rosenbrock_fun, n=n, bounds=bounds, maximize=True, verbose=True)
        print("Rosenbrock iters : x = {}, iters = {}".format(x, iters))

    # __________________________________________________________
    # FUNCTIONS FOR TESTING OPTIMIZATION
    # - Rastrigin function
    # - Sphere function
    # __________________________________________________________

    def tetris_fun(self, theta, samples=20, training=True, verbose=False):
        env = TetrisEnv(training=training)
        env.reset()
        fxs = samples*[0]
        for s in range(samples):
            fx = 0
            while True:
                state = env.state.copy()
                actions = env.get_actions()
                Q = len(actions)*[0]
                fs = np.zeros((len(actions),8))
                # print(actions)
                for i in range(len(actions)):
                    a_i = actions[i]
                    next_state, reward, done, _ = env.step(a_i)
                    fs[i,:] = features(state, reward, next_state, verbose=False)
                    Q[i] = theta.dot(fs[i,:])
                    env.set_state(state)
                V_min = np.min(Q)
                i_min = np.argmin(Q)
                a_min = actions[i_min]

                state, reward, done, _ = env.step(a_min)
                fx += reward
            
                if done:
                    if verbose: 
                        print("+==========================================+")
                        print("+                  TETRIS                  +")
                        print("+==========================================+\n")
                        print(f'Weights: {theta}')
                        env.render()
                        print(f'\nLines cleared: {state.cleared}')
                        print(f'Score: {fx}\n')
                    env.reset()
                    fxs[s] = fx
                    break
        fx_avg = np.mean(fxs)
        if verbose: 
            print(f'High score: {np.max(fxs)}')
            print(f'Average Score: {fx_avg}')
        return fx_avg
            

    def rastrigin_fun(self, x, samples=1, verbose=False):
        A = 10
        n = len(x)
        fx = A*n
        for i in range(n):
            fx += x[i]**2 - A*np.cos(2*pi*x[i])
        return fx

    def sphere_fun(self, x, samples=1, verbose=False):
        n = len(x)
        fx = 0
        for i in range(n):
            fx += x[i]**2
        return fx

    def rosenbrock_fun(self, x, samples=1, verbose=False):
        n = len(x)
        fx = 0
        for i in range(n-1):
            fx += 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return fx

    # __________________________________________________________
    # OPTIMIZATION FUNCTIONS
    # - Cross-Entropy Method
    # - Nelder-Mead (Simplex Search)
    # __________________________________________________________

    def cross_entropy_method(self, f, n, bounds, tol=1e-20, samples=25, function_samples=1, N_top=5, noise=NO_NOISE, 
                             maximize=False, normalize=False, training=False, verbose=False, checkpoints=False, load_save_state=False):
        max_iters = 1000

        save_state = {}
        iters = 0

        # gaussian means
        mu = np.zeros((n,))
        sigma = np.zeros((n,))

        # Load CEM save state from json file
        if load_save_state and os.path.exists('./cem_save_state.json'):
            with open('./cem_save_state.json') as of:
                save_state = json.load(of)
                iters = save_state['last_iters'] + 1
                mu = np.array(save_state[str(iters - 1)]['mu'])
                sigma = np.array(save_state[str(iters - 1)]['sigma'])
        else:
            for i in range(n):
                mu[i] = (bounds[i][0] + bounds[i][1])/2
                sigma[i] = (bounds[i][0] - bounds[i][1])/4

        x = np.zeros((samples, n))
        fx = np.zeros((samples,))

        while True:

            for s in range(samples):
                for i in range(n):
                    x[s,i] = random.gauss(mu[i], sigma[i])
                if normalize: x[s,:] = x[s,:]/np.linalg.norm(x[s,:])
                fx[s] = f(x[s], samples=function_samples)

            # Order according to values at vertices
            order = np.argsort(fx) # indices sorted lowest to highest
            if maximize:
                order = np.flip(order)
            
            x_top = np.zeros((N_top, n))
            fx_top = np.zeros((N_top,))
            for i in range(N_top):
                x_top[i] = x[order[i]]
                fx_top[i] = fx[order[i]]
                
            for i in range(n):
                mu[i] = np.mean(x_top[:,i])
                sigma2 = np.var(x_top[:,i])
                if noise == NO_NOISE:
                    sigma[i] = np.sqrt(sigma2)
                elif noise == DECREASING_NOISE:
                    sigma[i] = np.sqrt(sigma2 + max(0.05 - iters/1000, 0))
                else:
                    sigma[i] = np.sqrt(sigma2 + noise)
                    
            # Save CEM progress
            if checkpoints:
                save_state['last_iters'] = iters
                save_state[str(iters)] = {
                    'mu': mu.tolist(),
                    'sigma': sigma.tolist(),
                    'fx_top': fx_top.tolist()
                }

                with open('cem_save_state.json', 'w') as of:
                    json.dump(save_state, of)

            # Evaluate best vertex
            if verbose: 
                print(f'Iteration = {iters}, Deviation = {sigma.mean()}, Mean elite score = {fx_top.mean()}')
                print(f'Mean Elite Weights = {mu}')
            
            if iters >= max_iters or np.abs(sigma).sum() < tol:
                break
            
            iters += 1

        x_f = mu # Final weights
        fx_f = f(x_f) # Final average score

        return x_f, fx_f, iters

    def nelder_mead(self, f, n, bounds, tol=1e-12, maximize=False, normalize=False, training=False, verbose=False):
        # Initialize simplex
        x = np.zeros((n+1, n))
        for i in range(n):
            b_l, b_u = bounds[i][0], bounds[i][1]
            b_r, b_c = b_u - b_l, (b_l + b_u)/2 # range and center of bounds
            x[:,i] = b_r*(np.random.rand(n+1,)-0.5) + b_c

        iters = 0
        fx_f = None

        while True:
            # 1. Evaluate vertices of simplex
            fx = np.zeros((n+1,))
            for i in range(n+1):
                if normalize: x[i,:] = x[i,:]/np.linalg.norm(x[i,:])
                fx[i] = f(x[i,:])

            # std_dev = np.std(fx)
            std_dev = np.abs(np.std(x, axis=0)).sum()

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
            if maximize:
                order = np.flip(order)

            # Evaluate best vertex
            x_1 = x[order[0],:]
            fx_1 = fx[order[0]]
            if verbose: 
                if training:
                    f(x[order[0],:], training=False, verbose=verbose) # if verbose, show simulation
                else:
                    f(x[order[0],:], verbose=verbose) # if verbose, show simulation

            # Find worst vertex
            h = order[n]
            x_h = x[h,:]
            fx_h = fx[h]

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

            x_n = x[order[n-1],:]
            fx_n = f(x_n)

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

if __name__ == "__main__":
    learner = Learner()
    learner.learn()
    