import sys, os
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, os.path.abspath("envs"))
from tetris import TetrisEnv, TetrisState

def features(state, reward, next_state):
    f1, f2 = action_dependent_features(next_state, reward) 
    f3, f4, f5, f6, f7, f8 = action_independent_features(state)
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8])

def action_dependent_features(state, reward, verbose=False):
    f1 = 0
    n_rows, n_cols = state.field.shape
    for i in range(n_rows):
        if np.max(state.field[i,:]) == state.turn:
            break
        f1 += 1

    f2 = (4 - (state.field == state.turn).sum())*reward
            
    if verbose:
        print("Action-dependent features:")
        print("Next-Field:")
        print()
        print(state.field)
        print()
        print("(f1): Landing height")
        print("The height at which the current piece fell.")
        print()
        print(f'(f1) = {f1}')
        print()
        print("(f2): Eroded pieces")
        print("The contribution of the last piece to the cleared lines times the number of cleared lines.")
        print()
        print(f'(f2) = {f2}')
        print()
    return [f1, f2]

def action_independent_features(state, verbose=False):
    # Compute Preliminaries
    #   f3-f5: Binarize field and add 'borders' for computing 
    field = state.field.copy()
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

    if verbose:
        print("Action-independent features:")
        print()
        print("Bordered Field:")
        print()
        print(bordered_field)
        print()
        print("(f3): Row transitions")
        print("Number of filled cells adjacent to empty cells summed over all rows. Borders count.")
        print()
        print(f'(f3) = {f3}')
        print()
        print((delta_column != 0)+1-1)
        print()
        print("(f4): Column transitions")
        print("Number of filled cells adjacent to empty cells summed over all columns. Borders count as filled cells.")
        print()
        print((delta_row != 0)+1-1)
        print()
        print(f'(f4) = {f4}')
        print()
        print("(f5): Number of holes")
        print("The number of empty cells with at least one filled cell above.")
        print()
        print((delta_row == -1)+0)
        print()
        print(f'(f5) = {f5}')
        print()

    return [f3, f4, f5, f6, f7, f8]

if __name__=="__main__":
    env = TetrisEnv()
    env.reset()
    while True:

        state_pre = env.state.copy()

        env.render()

        actions = env.get_actions()
        action = actions[np.random.randint(len(actions))]
        state_post, reward, done, _ = env.step(action)

        env.render()
        fx = features(state_pre, reward, state_post)
        print(f'features: {fx}')

        if fx[1] > 0: break

        # input()
        print('#####################################################################')
        if done:
            env.reset()
            # break