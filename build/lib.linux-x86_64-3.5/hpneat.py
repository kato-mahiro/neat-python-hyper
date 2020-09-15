import sys,os
import math
import copy
import numpy as np

import tools

from update_weight import update_weight_for_abc #use cython module

class HyperNeat:
    def __init__(self, net, hpneat_config):

        self.hpneat_config = hpneat_config
        self.weight = np.zeros((self.hpneat_config.num_x * self.hpneat_config.num_y, self.hpneat_config.num_x * self.hpneat_config.num_y))
        self.bias = np.zeros((self.hpneat_config.num_x, self.hpneat_config.num_y))
        self.activate_val = np.zeros(( self.hpneat_config.num_x, self.hpneat_config.num_y ))

        self.set_initial_values(net)

    def set_initial_values(self, net):
        #set initial weight and bias
        xinpos= -1.0
        yinpos= -1.0
        xoutpos = -1.0
        youtpos = -1.0
        #XXX: 汚すぎる
        for xin in range(self.hpneat_config.num_x):
            yinpos = -1.0

            for yin in range(self.hpneat_config.num_y):
                xoutpos = -1.0
                self.bias[xin][yin] = net.activate([xinpos, yinpos, xinpos, yinpos])[1]

                for xout in range(self.hpneat_config.num_x):
                    youtpos = -1.0

                    for yout in range(self.hpneat_config.num_y):
                        self.weight[ tools._2d_to_1d(xin,yin) ][ tools._2d_to_1d(xout,yout) ] = \
                            net.activate([xinpos, yinpos, xoutpos, youtpos])[0] if abs(net.activate([xinpos, yinpos, xoutpos, youtpos])[0]) >= self.hpneat_config.weight_avail_theshold else 0.0

                        youtpos += 2.0 / (self.hpneat_config.num_y -1)
                    xoutpos += 2.0 / (self.hpneat_config.num_x -1)
                yinpos += 2.0 / (self.hpneat_config.num_x -1)
            xinpos += 2.0 / (self.hpneat_config.num_y -1)

    def activate(self, input_list):
        if( len(input_list) != len(self.hpneat_config.input_neuron_position) ):
            raise ValueError("argument ouf of range")

        for n in range(len(input_list)):
            self.activate_val[ self.hpneat_config.input_neuron_position[n][0] ][ self.hpneat_config.input_neuron_position[n][1] ] = input_list[n]

        activate_vec = tools.matrix_to_vector(self.activate_val)
        activate_vec = np.dot(activate_vec, self.weight)
        self.activate_val = tools.sigmoid_for_np_ndarray(tools.vector_to_matrix(activate_vec) + self.bias)

        output_vec = []
        for n in range(len(self.hpneat_config.output_neuron_position)):
            output_vec.append( self.activate_val[ self.hpneat_config.output_neuron_position[n][0] ][ self.hpneat_config.output_neuron_position[n][1] ])
        return output_vec

class HebbianABCModel:
    def __init__(self, net, hpneat_config, is_usecython=True):

        self.hpneat_config = hpneat_config

        self.is_usecython = is_usecython
        self.weight = np.zeros((self.hpneat_config.num_x * self.hpneat_config.num_y, self.hpneat_config.num_x * self.hpneat_config.num_y))
        self.A = np.zeros((self.hpneat_config.num_x * self.hpneat_config.num_y, self.hpneat_config.num_x * self.hpneat_config.num_y))
        self.B = np.zeros((self.hpneat_config.num_x * self.hpneat_config.num_y, self.hpneat_config.num_x * self.hpneat_config.num_y))
        self.C = np.zeros((self.hpneat_config.num_x * self.hpneat_config.num_y, self.hpneat_config.num_x * self.hpneat_config.num_y))
        self.ita = np.zeros((self.hpneat_config.num_x * self.hpneat_config.num_y, self.hpneat_config.num_x * self.hpneat_config.num_y))

        self.bias = np.zeros((self.hpneat_config.num_x, self.hpneat_config.num_y))
        self.activate_val = np.zeros(( self.hpneat_config.num_x, self.hpneat_config.num_y ))

        self.set_initial_values(net)

    def set_initial_values(self, net):
        #set initial weight and bias
        xinpos= -1.0
        yinpos= -1.0
        xoutpos = -1.0
        youtpos = -1.0
        #XXX: 汚すぎる
        for xin in range(self.hpneat_config.num_x):
            yinpos = -1.0

            for yin in range(self.hpneat_config.num_y):
                xoutpos = -1.0
                self.bias[xin][yin] = net.activate([xinpos, yinpos, xinpos, yinpos])[0]

                for xout in range(self.hpneat_config.num_x):
                    youtpos = -1.0

                    for yout in range(self.hpneat_config.num_y):
                        self.weight[ tools._2d_to_1d(xin,yin) ][ tools._2d_to_1d(xout,yout) ] = \
                            net.activate([xinpos, yinpos, xoutpos, youtpos])[1] if abs(net.activate([xinpos, yinpos, xoutpos, youtpos])[1]) >= self.hpneat_config.weight_avail_theshold else 0.0
                        self.A[ tools._2d_to_1d(xin,yin) ][ tools._2d_to_1d(xout,yout) ] =  net.activate([xinpos, yinpos, xoutpos, youtpos])[2]
                        self.B[ tools._2d_to_1d(xin,yin) ][ tools._2d_to_1d(xout,yout) ] =  net.activate([xinpos, yinpos, xoutpos, youtpos])[3]
                        self.C[ tools._2d_to_1d(xin,yin) ][ tools._2d_to_1d(xout,yout) ] =  net.activate([xinpos, yinpos, xoutpos, youtpos])[4]
                        self.ita[ tools._2d_to_1d(xin,yin) ][ tools._2d_to_1d(xout,yout) ] = net.activate([xinpos, yinpos, xoutpos, youtpos])[5]

                        youtpos += 2.0 / (self.hpneat_config.num_y -1)
                    xoutpos += 2.0 / (self.hpneat_config.num_x -1)
                yinpos += 2.0 / (self.hpneat_config.num_x -1)
            xinpos += 2.0 / (self.hpneat_config.num_y -1)

    def activate(self, input_list):

        if( len(input_list) != len(self.hpneat_config.input_neuron_position) ):
            raise ValueError("argument ouf of range")

        pre_activate_val = copy.deepcopy(self.activate_val)

        for n in range(len(input_list)):
            self.activate_val[ self.hpneat_config.input_neuron_position[n][0] ][ self.hpneat_config.input_neuron_position[n][1] ] = input_list[n]

        activate_vec = tools.matrix_to_vector(self.activate_val)
        activate_vec = np.dot(activate_vec, self.weight)
        self.activate_val = tools.sigmoid_for_np_ndarray(tools.vector_to_matrix(activate_vec) + self.bias)
        self.activate_val = self.activate_val.astype(np.float64)

        #update weight
        if(self.is_usecython):
            print(self.weight.dtype)
            print(self.activate_val.dtype)
            print(pre_activate_val.dtype)
            self.weight = update_weight_for_abc(self.hpneat_config.num_x, \
                                                self.hpneat_config.num_y, \
                                                self.weight, \
                                                pre_activate_val, \
                                                self.activate_val,\
                                                self.A, self.B, self.C, self.ita)

        else:
            for input_ in range( self.hpneat_config.num_x * self.hpneat_config.num_y):
                for output_ in range( self.hpneat_config.num_x * self.hpneat_config.num_y):
                    inputx = tools._1d_to_2d(input_)[0]
                    inputy = tools._1d_to_2d(input_)[1]
                    outputx = tools._1d_to_2d(output_)[0]
                    outputy = tools._1d_to_2d(output_)[1]

                    if(self.weight[input_][output_] != 0.0):
                        self.weight[input_][output_] += \
                            self.ita[input_][output_] * \
                            (
                                self.A[input_][output_] * pre_activate_val[inputx][inputy] * self.activate_val[outputx][outputy] + \
                                self.B[input_][output_] * pre_activate_val[inputx][inputy] +\
                                self.C[input_][output_] * self.activate_val[outputx][outputy]
                            )

        output_vec = []
        for n in range(len(self.hpneat_config.output_neuron_position)):
            output_vec.append( self.activate_val[ self.hpneat_config.output_neuron_position[n][0] ][ self.hpneat_config.output_neuron_position[n][1] ])
        return output_vec

