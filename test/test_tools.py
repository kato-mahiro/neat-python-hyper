import numpy as np
import neat
import hpneat
import tools

import hpneat_config

def test_read_setting():
    print(hpneat_config)
    print(hpneat_config.num_x)
    print(hpneat_config.num_y)

def test_2d_to_1d():
    hpneat_config.num_x = 5
    hpneat_config.num_y = 3
    assert tools._2d_to_1d(0,0) == 0
    assert tools._2d_to_1d(0,1) == 1
    assert tools._2d_to_1d(0,2) == 2
    assert tools._2d_to_1d(1,0) == 3
    assert tools._2d_to_1d(1,1) == 4
    assert tools._2d_to_1d(1,2) == 5

def test_1d_to_2d():
    hpneat_config.num_x = 5
    hpneat_config.num_y = 5
    assert tools._1d_to_2d(0) == (0,0)
    assert tools._1d_to_2d(1) == (0,1)
    assert tools._1d_to_2d(2) == (0,2)
    assert tools._1d_to_2d(3) == (0,3)
    assert tools._1d_to_2d(4) == (0,4)
    assert tools._1d_to_2d(5) == (1,0)

def test_matrix_to_vector():
    hpneat_config.num_x = 2
    hpneat_config.num_y = 3
    m1 = 20 * np.random.random((hpneat_config.num_x, hpneat_config.num_y)) -10
    print(m1)
    vec = tools.matrix_to_vector(m1)
    print(vec)
    print(vec.shape)
    m2=tools.vector_to_matrix(vec)
    print(m2)
    
    assert m1.shape == (2,3)
    assert vec.shape == (1,6)
    assert m1[1][0] == m2[1][0]
    assert m1.shape == m2.shape

def test_sigmoid_for_np_ndarray():
    m = np.zeros((2,2))
    m[0][0] = -1
    m[0][1] = 0
    m[1][0] = 1
    m[1][1] = 2
    print(" === test sigmoid === ")
    print(m)
    print(tools.sigmoid_for_np_ndarray(m))

if __name__=='__main__':
    test_read_setting()
    test_2d_to_1d()
    test_1d_to_2d()
    test_matrix_to_vector()
    test_sigmoid_for_np_ndarray()
