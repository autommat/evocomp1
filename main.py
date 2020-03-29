import math
import random
import xml.etree.ElementTree as ET

from deap import base
from deap import creator
from deap import tools


def prepare_toolbox(config):
    pass
#function-dependent
X_MIN, X_MAX = -10., 10.

#function independent
# prawdopodobieństwo krzyżowania
CXPB = 0.5
# prawdopodobieństwo mutacji
MUTPB = 0.2
#wielkość populacji
POP_SIZE = 30
# liczba pokoleń
NGEN = 100


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)

# toolbox.register("attr_float", random.random)
toolbox.register("attr_float", random.uniform, X_MIN, X_MAX)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, 2)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# definicja funkcji przystosowania do optymalizacji
def sum_of_squares(individual):
    x1 = individual[0]
    x2 = individual[1]
    y = x1**2 + x2**2
    return y,

def sin_times_cos(individual):
    x1 = individual[0]
    x2 = individual[1]
    y = math.sin(x1)*math.cos(x2)
    return y,

def sin_times_cos_plus_x1_plus_x2(individual):
    x1 = individual[0]
    x2 = individual[1]
    y = math.sin(x1)*math.cos(x2)+x1+x2
    return y,

# wybór funkcji przystosowania
toolbox.register("evaluate", sum_of_squares)

# wybór metody krzyżowania
toolbox.register("mate", tools.cxTwoPoint)


# 2 SPOSOBY MUTACJI
# indpb prawdopodobieństwo mutacji atrybutu/genu
# toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.05)




# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.

# wybór metody selekcji
toolbox.register("select", tools.selTournament, tournsize=2)
# toolbox.register("select", tools.selRoulette)

# ----------

# pop = toolbox.population(n=POP_SIZE)


def main(config_dict):
    # random.seed(64)

    best_individuals = []

    #populacja o rozmiarze POP_SIZE
    pop = toolbox.population(n=POP_SIZE)



    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    generation = 0

    # Begin the evolution
    # while max(fits) < 100 and generation < NGEN:
    while generation < NGEN:

        generation = generation + 1
        print(f"Pokolenie {generation}")

        # selekcja
        offspring = toolbox.select(pop, len(pop))


        #klonowanie osobników do nowej listy
        offspring = list(map(toolbox.clone, offspring))

        # krzyżowanie
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values
        #mutacja
        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        best_ind = tools.selBest(pop, 1)
        best_individuals.append(best_ind) #todo: add also the eval

    print("Koniec ewolucji")

    #wybór najlepszego osobnika z końcowej pupulacji
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    print(best_individuals)

if __name__ == "__main__":
    doc =ET.parse("config.xml")
    root = doc.getroot()
    config_dict={}
    config_dict["CXPB"] = float(root.find('CXPB').text)
    config_dict["MUTPB"] = float(root.find('MUTPB').text)
    config_dict["POP_SIZE"] = float(root.find('POP_SIZE').text)
    config_dict["NGEN"] = float(root.find('NGEN').text)
    mutation ={}
    mutation["name"] = root.find('MUTATION').text
    for attr, val in root.find('MUTATION').attrib.items():
        mutation[attr] = val
    config_dict["MUTATION"] = mutation
    main(config_dict)