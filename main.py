import random
import xml.etree.ElementTree as ET
import numpy
from deap import base, algorithms
from deap import creator
from deap import tools
from functions import *


#function-dependent todo: usunąć ograniczenia lub wczytać z config_dict
X_MIN, X_MAX = -10., 10.


def prepare_toolbox(config_dict):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.uniform, X_MIN, X_MAX)

    # osobnik to lista attr_float o długości 2
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, 2)

    # populacja to lista osobników
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # wybór funkcji przystosowania #todo: wybór funkcji przystosowania na podstawie config_dict
    toolbox.register("evaluate", sin_times_cos)

    # 2 SPOSOBY KRZYŻOWANIA: dwupunktowe i równomierne todo: wybór metody na podstawie config_dict
    # wybór metody krzyżowania
    # toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mate", tools.cxUniform, indpb=0.05)

    # 2 SPOSOBY MUTACJI todo: wybór metody na podstawie config_dict
    # indpb prawdopodobieństwo mutacji atrybutu/genu
    # toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) #todo: dodać metodę mutacji inną niż flipbit
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)

    # wybór metody selekcji
    toolbox.register("select", tools.selTournament, tournsize=2)

    return toolbox


def main(config_dict, toolbox):
    best_individuals = []

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
        best_individuals.append(best_ind) #todo: add also the eval

    print("Koniec ewolucji")

    #wybór najlepszego osobnika z końcowej pupulacji
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    print(best_individuals)


def prepare_statistics(): #todo: wyświetlić najlepszego osobnika matpplotlib.pyplot
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    return stats


if __name__ == "__main__":
    try:
        doc =ET.parse("config.xml")
        root = doc.getroot()

        algorithm = root.find('ALGORITHM').text

        config_dict={}
        config_dict["CXPB"] = float(root.find('CXPB').text)
        config_dict["MUTPB"] = float(root.find('MUTPB').text)
        config_dict["POP_SIZE"] = int(root.find('POP_SIZE').text)
        config_dict["NGEN"] = int(root.find('NGEN').text)
        config_dict["MIN"] = float(root.find('RANGE').find('MIN').text)
        config_dict["MAX"] = float(root.find('RANGE').find('MAX').text)
        mutation ={}
        mutation["name"] = root.find('MUTATION').text
        for attr, val in root.find('MUTATION').attrib.items():
            mutation[attr] = val
        config_dict["MUTATION"] = mutation
        toolbox = prepare_toolbox(config_dict)

        pop = toolbox.population(n=config_dict['POP_SIZE'])
        stats = prepare_statistics()

        rpop, logbook = None, None
        if algorithm == "Simple":
            rpop, logbook = algorithms.eaSimple(pop, toolbox, config_dict['CXPB'], config_dict['MUTPB'], config_dict['NGEN'], stats=stats)
        elif algorithm == "MuPlusLambda":
            mu = 20
            lambda_=7*mu
            rpop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu, lambda_, config_dict['CXPB'], config_dict['MUTPB'], config_dict['NGEN'], stats=stats)
        elif algorithm == "CustomSimple":    #todo:usunąć jeśli niepotrzebne
            main(config_dict, toolbox)
        else:
            print(f"algorytm o nazwie {algorithm} nie jest obsługiwany")

        gen = logbook.select("gen")
        fit_max = logbook.chapters["fitness"].select("max") #todo: wyświetlić matplotlib.pyplot

    except ET.ParseError:
        print("dokument xml jest źle sformatowany")
    except AttributeError:
        print("dokument xml ma nieprawidłowe elementy")
