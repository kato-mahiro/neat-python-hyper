import numpy as np
import neat
import hpneat

def test_2d_to_1d():
    hpneat.hpneat_config.num_x = 5
    hpneat.hpneat_config.num_y = 3
    assert hpneat._2d_to_1d(0,0) == 0
    assert hpneat._2d_to_1d(0,1) == 1
    assert hpneat._2d_to_1d(0,2) == 2
    assert hpneat._2d_to_1d(1,0) == 3
    assert hpneat._2d_to_1d(1,1) == 4
    assert hpneat._2d_to_1d(1,2) == 5

def test_1d_to_2d():
    hpneat.hpneat_config.num_x = 5
    hpneat.hpneat_config.num_y = 5
    assert hpneat._1d_to_2d(0) == (0,0)
    assert hpneat._1d_to_2d(1) == (0,1)
    assert hpneat._1d_to_2d(2) == (0,2)
    assert hpneat._1d_to_2d(3) == (0,3)
    assert hpneat._1d_to_2d(4) == (0,4)
    assert hpneat._1d_to_2d(5) == (1,0)

def test_matrix_to_vector():
    hpneat.hpneat_config.num_x = 2
    hpneat.hpneat_config.num_y = 3
    m1 = 20 * np.random.random((hpneat.hpneat_config.num_x, hpneat.hpneat_config.num_y)) -10
    print(m1)
    vec = hpneat.matrix_to_vector(m1)
    print(vec)
    print(vec.shape)
    m2=hpneat.vector_to_matrix(vec)
    print(m2)
    
    assert m1.shape == (2,3)
    assert vec.shape == (1,6)
    assert m1[1][0] == m2[1][0]
    assert m1.shape == m2.shape

if __name__=='__main__':
    test_2d_to_1d()
    test_1d_to_2d()
    test_matrix_to_vector()
