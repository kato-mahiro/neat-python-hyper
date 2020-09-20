import math
import numpy as np
import hpneat_config

def _2d_to_1d(x:int,y:int) -> int:
    if(x > hpneat_config.num_x-1 or x < 0 or y > hpneat_config.num_y-1 or y < 0):
        raise ValueError("argument ouf of range")
    return (hpneat_config.num_y * x)  + y

def _1d_to_2d(xy:int) -> (int,int):
    x = xy // hpneat_config.num_y
    y = xy - (x * hpneat_config.num_y)
    return x,y

def matrix_to_vector(mx):
    return np.reshape(mx, (1, mx.size ) )

def vector_to_matrix(vec):
    return np.reshape(vec, (hpneat_config.num_x, hpneat_config.num_y) )

def sigmoid_for_np_ndarray(np_ndarray):
    cut = lambda x:-100 if x < -100 else x
    np_cut = np.frompyfunc(cut, 1, 1)

    sigmoid = lambda x: 1 / (1+math.e ** -x) #sigmoid func
    np_sigmoid = np.frompyfunc(sigmoid, 1, 1)
    return np_sigmoid(np_cut(np_ndarray)).astype(np.float64)
