import numpy as np
from scipy.optimize import linear_sum_assignment

def match_players(features_a, features_b):
    ids_a = list(features_a.keys())
    ids_b = list(features_b.keys())
    mat = np.zeros((len(ids_a), len(ids_b)))
    for i, ida in enumerate(ids_a):
        for j, idb in enumerate(ids_b):
            f1 = features_a[ida]
            f2 = features_b[idb]
            sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
            mat[i, j] = 1 - sim
    row_ind, col_ind = linear_sum_assignment(mat)
    mapping = {ids_b[j]: ids_a[i] for i, j in zip(row_ind, col_ind)}
    return mapping