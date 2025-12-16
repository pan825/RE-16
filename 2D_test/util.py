import os
import time
import numpy as np
import matplotlib.pyplot as plt

F = 1/3
_WEIGHT_MATRIX_SYMMETRY = np.array([
    [F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, F, F],
    [F, F, F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, F, F, F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, F, F, F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, F, F, 0, 0, F, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, F, F, F, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, F, F, F, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, F, F, F, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, F, F, F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, F, F, F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, F, F, F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, F, 0, 0, F, F, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, F, F, F, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, F, F, F, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, F, F, F],
    [F, F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, F]
], dtype=np.float32)

def process_data_3D(data):
    '''
    input: data: (16, 16, 16, T)
    output: out: (16, 16, 16, T)
    '''
    W = _WEIGHT_MATRIX_SYMMETRY.astype(np.float32)
    C = np.zeros((16, 18), dtype=W.dtype)
    C[np.arange(8), np.arange(8)] = 1
    C[np.arange(8, 16), np.arange(10, 18)] = 1
    L = C @ W.T @ C.T  # (16, 16)
    R = C @ W   @ C.T  # (16, 16)
    Z = C @ W   @ C.T  # (16, 16)
    out = np.einsum('ai,ijkt,jb,kc->abct', L, data, R, Z, optimize=True)  # (16, 16, 16, T)

    return out

def process_data(data):
    '''
    input: data: (16, 16, T)
    output: out: (16, 16, T)
    '''
    W = _WEIGHT_MATRIX_SYMMETRY.astype(np.float32)
    C = np.zeros((16, 18), dtype=W.dtype)
    C[np.arange(8), np.arange(8)] = 1
    C[np.arange(8, 16), np.arange(10, 18)] = 1
    L = C @ W.T @ C.T   # (16, 16)
    R = C @ W   @ C.T  # (16, 16)
    out = np.einsum('ai, ijt, jb -> abt', L, data, R, optimize=True)  # (16, 16, T) 
    return out


