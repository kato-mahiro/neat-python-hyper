import math
import numpy as np

import hpneat_config
import tools

class HyperNeat:
    def __init__(self, net):

        self.weight = np.zeros((hpneat_config.num_x * hpneat_config.num_y, hpneat_config.num_x * hpneat_config.num_y))
        self.bias = np.zeros((hpneat_config.num_x, hpneat_config.num_y))
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
                self.bias[xin][yin] = net.activate([xinpos, yinpos, xinpos, yinpos])[1]

                for xout in range(hpneat_config.num_x):
                    youtpos = -1.0

                    for yout in range(hpneat_config.num_y):
                        self.weight[ tools._2d_to_1d(xin,yin) ][ tools._2d_to_1d(xout,yout) ] = \
                            net.activate([xinpos, yinpos, xoutpos, youtpos])[0] if abs(net.activate([xinpos, yinpos, xoutpos, youtpos])[0]) >= hpneat_config.weight_avail_theshold else 0.0

                        youtpos += 2.0 / (hpneat_config.num_y -1)
                    xoutpos += 2.0 / (hpneat_config.num_x -1)
                yinpos += 2.0 / (hpneat_config.num_x -1)
            xinpos += 2.0 / (hpneat_config.num_y -1)

    def activate(self, input_list):
        if( len(input_list) != len(hpneat_config.input_neuron_position) ):
            raise ValueError("argument ouf of range")

        for n in range(len(input_list)):
            self.activate_val[ hpneat_config.input_neuron_position[n][0] ][ hpneat_config.input_neuron_position[n][1] ] = input_list[n]

        activate_vec = tools.matrix_to_vector(self.activate_val)
        activate_vec = np.dot(activate_vec, self.weight)
        self.activate_val = tools.sigmoid_for_np_ndarray(tools.vector_to_matrix(activate_vec) + self.bias)

        output_vec = []
        for n in range(len(hpneat_config.output_neuron_position)):
            output_vec.append( self.activate_val[ hpneat_config.output_neuron_position[n][0] ][ hpneat_config.output_neuron_position[n][1] ])
        return output_vec

