import neat

import hpneat

config = neat.Config(neat.DefaultGenome,
                     neat.DefaultReproduction,
                     neat.DefaultSpeciesSet,
                     neat.DefaultStagnation,
                     './neat_config.ini')

#prepare genom
p = neat.Population(config)
plist = list(p.population.items())
g = plist[0][1]

def test_HyperNeat():
    #create hyperneat object
    net = neat.nn.FeedForwardNetwork.create(g,config)
    hyperneat = hpneat.HyperNeat(net)
    print(hyperneat.weight)

    i=[1,2]
    hyperneat.activate(i)
    print(hyperneat.activate_val)

if __name__=='__main__':
    test_HyperNeat()
