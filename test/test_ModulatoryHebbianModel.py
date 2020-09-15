import time
import neat
import hpneat

import hpneat_config

def test_ModulatoryHebbianModel():
    hpneat_config.num_x = 5
    hpneat_config.num_y = 5
    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         './neat_config.ini')
    #prepare genom
    p = neat.Population(config)
    plist = list(p.population.items())
    g = plist[0][1]
    print(type(g))
    assert type(g).__name__ == 'DefaultGenome'
    net = neat.nn.FeedForwardNetwork.create(g, config)
    my_mhmodel = hpneat.ModulatoryHebbianModel(net, hpneat_config)

    #assert hyperneat parameters
    print("~~~~~~~~~~~~~~~~~")
    print("weight")
    print(my_mhmodel.weight.shape)
    print(my_mhmodel.weight)
    print("A")
    print(my_mhmodel.A.shape)
    print(my_mhmodel.A)
    print("B")
    print(my_mhmodel.B.shape)
    print(my_mhmodel.B)
    print("C")
    print(my_mhmodel.C.shape)
    print(my_mhmodel.C)
    print("ita")
    print(my_mhmodel.ita.shape)
    print(my_mhmodel.ita)
    print("bias")
    print(my_mhmodel.bias.shape)
    print(my_mhmodel.bias)
    print("activate_val")
    print(my_mhmodel.activate_val.shape)
    print(my_mhmodel.activate_val)
    print("modulated_val")
    print(my_mhmodel.modulated_val.shape)
    print(my_mhmodel.modulated_val)
    print("~~~~~~~~~~~~~~~~~")

    print('output:',my_mhmodel.activate([1,2]))
    print(my_mhmodel.weight)
    print('output:',my_mhmodel.activate([1,1]))
    print(my_mhmodel.weight)
    start = time.time()
    print('output:',my_mhmodel.activate([0,0]))
    end = time.time() - start
    print(my_mhmodel.weight)

    print("サイズ", hpneat_config.num_x, hpneat_config.num_y,"で1更新にかかる時間は", end ,"[sec]")
    print("100個体，100ステップの実験なら1世代あたり",end * 10000,"[sec]")

if __name__=='__main__':
    test_ModulatoryHebbianModel()
