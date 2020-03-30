from random import random

from deap import tools


def main(config_dict, toolbox, stats=None):

    #populacja o rozmiarze POP_SIZE
    pop = toolbox.population(n=config_dict['POP_SIZE'])



    print("Początek ewolucji")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print(f"liczebność pierwszej populacji: {len(pop)}")

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # numer pokolenia
    generation = 0

    # Begin the evolution
    # while max(fits) < 100 and generation < NGEN:
    while generation < config_dict["NGEN"]:

        generation = generation + 1
        print(f"Pokolenie {generation}")

        # selekcja
        offspring = toolbox.select(pop, len(pop))


        #klonowanie osobników przed krzyżowaniem i mutacją
        #https://deap.readthedocs.io/en/master/tutorials/basic/part2.html#mutation
        offspring = list(map(toolbox.clone, offspring))

        # krzyżowanie
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < config_dict["CXPB"]:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values
        #mutacja
        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < config_dict["MUTPB"]:
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
        if stats:
            record = stats.compile(pop)

    print("Koniec ewolucji")

    #wybór najlepszego osobnika z końcowej pupulacji
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    return pop, stats