import neat
import hpneat

def test_HyperNeat():
    hpneat.hpneat_config.num_x = 3
    hpneat.hpneat_config.num_y = 4
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
    my_hyper_neat = hpneat.HyperNeat(net)

    #assert hyperneat parameters
    print("~~~~~~~~~~~~~~~~~")
    print("weight")
    print(my_hyper_neat.weight.shape)
    print(my_hyper_neat.weight)
    print("bias")
    print(my_hyper_neat.bias.shape)
    print(my_hyper_neat.bias)
    print("activate_val")
    print(my_hyper_neat.activate_val.shape)
    print(my_hyper_neat.activate_val)
    print("~~~~~~~~~~~~~~~~~")

    my_hyper_neat.activate([1,2])
    my_hyper_neat.activate([1,1])
    my_hyper_neat.activate([0,0])

if __name__=='__main__':
    test_HyperNeat()
