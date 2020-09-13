from copy import deepcopy
import time
import neat
import hpneat

import hpneat_config

def test_HebbianABCModel():
    hpneat_config.num_x = 2
    hpneat_config.num_y = 2
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
    my_abc_model = hpneat.HebbianABCModel(net, hpneat_config)
    my_abc_model2 = deepcopy(my_abc_model)
    assert my_abc_model is not my_abc_model2
    my_abc_model2.is_usecython = False

    my_abc_model.activate([1,2])
    my_abc_model.activate([1,1])
    my_abc_model.activate([0,0])

    my_abc_model2.activate([1,2])
    my_abc_model2.activate([1,1])
    my_abc_model2.activate([0,0])

    print(my_abc_model.is_usecython)
    print(my_abc_model.weight)
    print(my_abc_model2.is_usecython)
    print(my_abc_model2.weight)

if __name__=='__main__':
    test_HebbianABCModel()
