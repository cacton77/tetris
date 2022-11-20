#!/usr/bin/env python
import sys, os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('classic')

save_state = {}

if os.path.exists('./cem_save_state.json'):
    with open('./cem_save_state.json') as of:
        save_state = json.load(of)
else:
    sys.exit("No optimization history.")

last_iters = save_state['last_iters']

N_top = len(save_state['0']['fx_top'])
n = len(save_state['0']['mu'])

iters = np.zeros((last_iters + 1,))
fx_top = np.zeros((last_iters + 1, N_top))
fx_top_mean = np.zeros((last_iters + 1,))
mu = np.zeros((last_iters + 1, n))

# Unpack data from optimization into local variables
for it in range(last_iters + 1):
    iters[it] = it
    fx_top[it,:] = save_state[str(it)]['fx_top']
    fx_top_mean[it] = np.mean(save_state[str(it)]['fx_top'])
    mu[it,:] = save_state[str(it)]['mu']

# Elite performance

fig, ax = plt.subplots()

plt.plot(iters, fx_top_mean, marker='s')
for i in range(N_top):
    plt.scatter(iters, fx_top[:,i], marker='o', s=8)
plt.title("Average Elite Performance During Training")
plt.grid()
ax.set_xlim(0, last_iters)
ax.set_xlabel("Iteration")
ax.set_ylabel("Score")

plt.show()

# Weight (mu) convergence

fig, ax = plt.subplots()

for i in range(n):
    plt.plot(iters, mu[:,i])
plt.grid()
ax.set_xlim(0, last_iters)
ax.set_xlabel("Iteration")
ax.set_ylabel("Score")
ax.legend(['w_1','w_2','w_3','w_4','w_5','w_6','w_7','w_8',])

plt.show()