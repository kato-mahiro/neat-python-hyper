import time
import neat
import hpneat

def test_HebbianABCModel():
    hpneat.hpneat_config.num_x = 10
    hpneat.hpneat_config.num_y = 10
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
    my_abc_model = hpneat.HebbianABCModel(net)

    #assert hyperneat parameters
    print("~~~~~~~~~~~~~~~~~")
    print("weight")
    print(my_abc_model.weight.shape)
    print(my_abc_model.weight)
    print("A")
    print(my_abc_model.A.shape)
    print(my_abc_model.A)
    print("B")
    print(my_abc_model.B.shape)
    print(my_abc_model.B)
    print("C")
    print(my_abc_model.C.shape)
    print(my_abc_model.C)
    print("ita")
    print(my_abc_model.ita.shape)
    print(my_abc_model.ita)
    print("bias")
    print(my_abc_model.bias.shape)
    print(my_abc_model.bias)
    print("activate_val")
    print(my_abc_model.activate_val.shape)
    print(my_abc_model.activate_val)
    print("~~~~~~~~~~~~~~~~~")

    print('output:',my_abc_model.activate([1,2]))
    print(my_abc_model.weight)
    print('output:',my_abc_model.activate([1,1]))
    print(my_abc_model.weight)
    start = time.time()
    print('output:',my_abc_model.activate([0,0]))
    end = time.time() - start
    print(my_abc_model.weight)

    print("サイズ", hpneat.hpneat_config.num_x, hpneat.hpneat_config.num_y,"で1更新にかかる時間は", end ,"[sec]")
    print("100個体，100ステップの実験なら1世代あたり",end * 10000,"[sec]")

if __name__=='__main__':
    test_HebbianABCModel()
