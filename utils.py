import numpy as np
from scipy.signal import find_peaks

def action_independent_features(field):
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