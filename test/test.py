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
