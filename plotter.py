#!/usr/bin/env python
import sys, os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.style.use('classic')

save_state = {}

file_name = 'cem_save_state_not_norm.json'
file_name = 'cem_save_state.json'

if os.path.exists(file_name):
    with open(file_name) as of:
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
sigma = np.zeros((last_iters + 1, n))

# Unpack data from optimization into local variables
for it in range(last_iters + 1):
    iters[it] = it
    fx_top[it,:] = save_state[str(it)]['fx_top']
    fx_top_mean[it] = np.mean(save_state[str(it)]['fx_top'])
    mu[it,:] = save_state[str(it)]['mu']
    sigma[it,:] = save_state[str(it)]['sigma']

# Elite performance

fig, ax = plt.subplots()

plt.figure(1)
plt.errorbar(iters, fx_top_mean, yerr=np.std(fx_top, axis=1), marker='s')
for i in range(N_top):
    plt.scatter(iters, fx_top[:,i], marker='8', s=8)
plt.title("Average Elite Performance During Training")
plt.grid()
ax.set_xlim(0, last_iters)
ax.set_ylim(0, 1.2*np.max(fx_top_mean))
ax.set_xlabel("Iteration")
ax.set_ylabel("Score")

# plt.show()

# Weight (mu) convergence

fig, ax1 = plt.subplots()

cmap = cm.get_cmap('viridis')
# cmap = cm.get_cmap('twilight_shifted_r')
for i in range(n):
    # plt.errorbar(iters, mu[:,i], yerr=sigma[:,i]/5)
    w_upper = mu[:,i]+sigma[:,i]
    w_lower = mu[:,i]-sigma[:,i]
    plt.figure(2)
    plt.plot(iters, mu[:,i], lw=2, marker='s', c=cmap(i/(n-1)), label=f'weight {i+1}')
    plt.fill_between(iters, w_upper, w_lower, color=cmap(i/(n-1)) ,alpha=0.4)
plt.set_cmap('viridis')
plt.grid()
ax1.set_xlim(0, last_iters)
# ax1.set_ylim(-1, 1)
ax1.set_xlabel("Iteration")
ax1.legend(loc='lower right')
# ax.legend(['w_1','w_2','w_3','w_4','w_5','w_6','w_7','w_8',])

# for i in range(n):
#     it_l = 80
#     w = i
#     w_upper = mu[it_l:-1,w]+sigma[it_l:-1,w]
#     w_lower = mu[it_l:-1,w]-sigma[it_l:-1,w]
#     plt.figure(3 + i)
#     plt.plot(iters[it_l:-1], mu[it_l:-1,w],  lw=2, marker='s', c=cmap(w/(n-1)), label=f'weight {i+1}')
#     plt.fill_between(iters[it_l:-1], w_upper, w_lower, color=cmap(w/(n-1)) ,alpha=0.4)
#     plt.xlabel('Iteration')
#     plt.title(f'Weight {i+1}')
#     plt.grid()

plt.figure(3)
print("Format: (avg lines, avg score, avg time)")
performance = np.array([[1.42493000e+04, 1.43197000e+06, 9.35003226e+01],
       [5.63130000e+03, 5.65140000e+05, 3.75441704e+01],
       [1.19808000e+04, 1.20608000e+06, 8.15985394e+01],
       [3.40380000e+03, 3.41940000e+05, 2.45650228e+01],
       [1.77632000e+04, 1.78761000e+06, 1.26963580e+02],
       [1.35188000e+04, 1.35916000e+06, 9.65734191e+01],
       [9.97030000e+03, 1.00375000e+06, 6.75112182e+01],
       [5.52330000e+03, 5.56810000e+05, 3.56535454e+01],
       [1.16090000e+04, 1.16586000e+06, 7.44274133e+01],
       [2.41988000e+04, 2.43392000e+06, 1.53574345e+02],
       [1.12983000e+04, 1.13791000e+06, 2.20103107e+03],
       [4.27315000e+04, 4.32805000e+06, 2.77165023e+02],
       [7.95740000e+03, 8.00060000e+05, 5.09376708e+01],
       [5.06340000e+03, 5.09950000e+05, 3.26052265e+01],
       [3.25541000e+04, 3.29482000e+06, 2.07884968e+02],
       [2.50172000e+04, 2.52764000e+06, 1.60291789e+02],
       [2.68879000e+04, 2.73175000e+06, 1.73169932e+02],
       [8.03330000e+03, 8.06330000e+05, 5.18682212e+01],
       [5.79960000e+04, 5.85468000e+06, 3.70531614e+02],
       [3.05662000e+04, 3.11962000e+06, 1.94919471e+02],
       [3.55265000e+04, 3.59052000e+06, 2.27821033e+02],
       [4.31187000e+04, 4.37027000e+06, 2.76012126e+02],
       [1.92799000e+04, 1.94077000e+06, 1.23409180e+02],
       [3.90218000e+04, 3.96246000e+06, 2.50204994e+02],
       [2.26411000e+04, 2.28719000e+06, 1.44614847e+02],
       [4.67757000e+04, 4.76797000e+06, 2.99211730e+02],
       [1.82394000e+04, 1.84806000e+06, 1.17457106e+02],
       [3.27704000e+04, 3.30297000e+06, 2.09268579e+02],
       [3.44809000e+04, 3.48201000e+06, 2.18094326e+02],
       [3.83212000e+04, 3.90058000e+06, 2.45002809e+02],
       [2.37549000e+04, 2.39965000e+06, 1.50462360e+02],
       [4.35775000e+04, 4.45624000e+06, 2.89166685e+02],
       [2.40622000e+04, 2.45243000e+06, 1.56164931e+02],
       [2.34909000e+04, 2.37414000e+06, 1.56369416e+02],
       [1.02623000e+04, 1.03335000e+06, 6.81666365e+01],
       [1.46861000e+04, 1.49233000e+06, 9.76949129e+01],
       [3.80287000e+04, 3.86908000e+06, 2.40890101e+02],
       [4.16919000e+04, 4.27163000e+06, 2.61021129e+02],
       [3.86735000e+04, 3.89243000e+06, 2.43748299e+02],
       [5.92732000e+04, 6.01308000e+06, 3.72176939e+02],
       [3.73599000e+04, 3.79864000e+06, 2.34390347e+02],
       [2.73880000e+03, 2.76240000e+05, 1.74055812e+01],
       [7.16460000e+03, 7.26350000e+05, 4.51575082e+01],
       [9.66050000e+03, 9.80770000e+05, 6.07481202e+01],
       [5.42757000e+04, 5.50939000e+06, 3.40975859e+02],
       [3.36506000e+04, 3.43598000e+06, 2.11867185e+02],
       [5.25729000e+04, 5.35582000e+06, 3.29656533e+02],
       [3.34372000e+04, 3.39160000e+06, 2.12922621e+02],
       [3.17944000e+04, 3.23661000e+06, 1.99974261e+02],
       [4.30738000e+04, 4.38646000e+06, 2.74641885e+02]])

plt.plot(range(50),performance[:,0])
plt.title('Tetris Player Performance')
plt.xlabel('Iteration')
plt.ylabel('Lines Cleared')
plt.grid()

plt.show()