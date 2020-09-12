import cython
import numpy as np
cimport numpy as cnp

def c1d_to_2d(int xy, int num_x, int num_y):
    """cython用の _1d_to_2d関数
    cython内でのみ使用
    """
    cdef:
        int x, y
    x = xy // num_y
    y = xy - (x * num_y)
    return x,y

# pythonから呼び出す関数
def update_weight_for_abc(int num_x, int num_y, cnp.ndarray weight, cnp.ndarray pre_activate_val, cnp.ndarray activate_val,\
                              cnp.ndarray A, cnp.ndarray B, cnp.ndarray C, cnp.ndarray ita ):
    cdef:
        int input_, output_, inputx, inputy, outputx, outputy
    for input_ in range(num_x * num_y):
        for output_ in range(num_x * num_y):
            inputx = c1d_to_2d(input_, num_x, num_y)[0]
            inputy = c1d_to_2d(input_, num_x, num_y)[1]
            outputx = c1d_to_2d(output_, num_x, num_y)[0]
            outputy = c1d_to_2d(output_, num_x, num_y)[1]

            if(weight[input_][output_] != 0.0):
                weight[input_][output_] += \
                    ita[input_][output_] * \
                    (
                        A[input_][output_] * pre_activate_val[inputx][inputy] * activate_val[outputx][outputy] + \
                        B[input_][output_] * pre_activate_val[inputx][inputy] +\
                        C[input_][output_] * activate_val[outputx][outputy]
                    )

    return weight
