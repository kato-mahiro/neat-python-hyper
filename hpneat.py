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

class HyperNeat:
    def __init__(self, net):
        self.weight = np.zeros((hpneat_config.num_x * hpneat_config.num_y, hpneat_config.num_x * hpneat_config.num_y))
        self.bias = np.zeros((hpneat_config.num_x * hpneat_config.num_y, hpneat_config.num_x * hpneat_config.num_y))
        self.activate_val = np.zeros(( hpneat_config.num_x, hpneat_config.num_y ))

        #set initial weight and bias
        xinpos= -1.0
        yinpos= -1.0
        xoutpos = -1.0
        youtpos = -1.0
        #XXX: 汚すぎる
        for xin in range(hpneat_config.num_x):
            yinpos = -1.0

            for yin in range(hpneat_config.num_y):
                xoutpos = -1.0

                for xout in range(hpneat_config.num_x):
                    youtpos = -1.0

                    for yout in range(hpneat_config.num_y):
                        self.weight[ _2d_to_1d(xin,yin) ][ _2d_to_1d(xout,yout) ] = \
                            net.activate([xinpos, yinpos, xoutpos, youtpos])[0] if abs(net.activate([xinpos, yinpos, xoutpos, youtpos])[0]) >= hpneat_config.weight_avail_theshold else None
                        self.bias[ _2d_to_1d(xin,yin) ][ _2d_to_1d(xout,yout) ] = net.activate([xinpos, yinpos, xoutpos, youtpos])[1] 

                        youtpos += 2.0 / (hpneat_config.num_y -1)
                    xoutpos += 2.0 / (hpneat_config.num_x -1)
                yinpos += 2.0 / (hpneat_config.num_x -1)
            xinpos += 2.0 / (hpneat_config.num_y -1)
