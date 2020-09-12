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

class HyperNeat:
    def __init__(self, net):

        self.weight = np.zeros((hpneat_config.num_x * hpneat_config.num_y, hpneat_config.num_x * hpneat_config.num_y))
        self.bias = np.zeros((hpneat_config.num_x * hpneat_config.num_y, hpneat_config.num_x * hpneat_config.num_y))
        self.activate_val = np.zeros(( hpneat_config.num_x, hpneat_config.num_y ))

        self.set_initial_values(net)

    def set_initial_values(self, net):
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
                            net.activate([xinpos, yinpos, xoutpos, youtpos])[0] if abs(net.activate([xinpos, yinpos, xoutpos, youtpos])[0]) >= hpneat_config.weight_avail_theshold else 0.0
                        self.bias[ _2d_to_1d(xin,yin) ][ _2d_to_1d(xout,yout) ] = net.activate([xinpos, yinpos, xoutpos, youtpos])[1] 

                        youtpos += 2.0 / (hpneat_config.num_y -1)
                    xoutpos += 2.0 / (hpneat_config.num_x -1)
                yinpos += 2.0 / (hpneat_config.num_x -1)
            xinpos += 2.0 / (hpneat_config.num_y -1)

    def activate(self, input_list):
        if( len(input_list) != len(hpneat_config.input_neuron_position) ):
            raise ValueError("argument ouf of range")
        for n in range(len(input_list)):
            self.activate_val[ hpneat_config.input_neuron_position[n][0] ][ hpneat_config.input_neuron_position[n][1] ] = input_list[n]
        print(self.activate_val)
        activate_vec = matrix_to_vector(self.activate_val)
        activate_vec = np.dot(activate_vec, self.weight)
        self.activate_val = vector_to_matrix(activate_vec)
        print(self.activate_val)
        output_vec = []
        for n in range(len(hpneat_config.output_neuron_position)):
            output_vec.append( self.activate_val[ hpneat_config.output_neuron_position[n][0] ][ hpneat_config.output_neuron_position[n][1] ])
        print(output_vec)
        return output_vec

