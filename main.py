import random
from os.path import exists
import numpy
from deap import base, algorithms
from deap import creator
from deap import tools
from custom_evolutionary_algorithms import main
from functions import *
import matplotlib.pyplot as plt
from xmlparser import XmlParser, XmlParserError


def prepare_toolbox(config_dict):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # typ i zakres genu osobnika
    toolbox.register("attr_float", random.uniform, config_dict["MIN"], config_dict["MAX"])

    # osobnik to lista attr_float o długości 2
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 2)

    # populacja to lista osobników
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # wybór funkcji przystosowania #todo: wybór funkcji przystosowania na podstawie config_dict
    toolbox.register("evaluate", sin_times_cos)

    #todo: dodać funkcję kary

    # wybór metody krzyżowania
    if config_dict["CROSSOVER"]["name"] == "cxTwoPoint":
        toolbox.register("mate", tools.cxTwoPoint)
    elif config_dict["CROSSOVER"]["name"] == "cxUniform":
        indpb = config_dict["CROSSOVER"]["indpb"]
        toolbox.register("mate", tools.cxUniform, indpb=indpb)
    else:
        raise NotImplementedError

    #wybór metody mutacji
    # indpb prawdopodobieństwo mutacji atrybutu/genu
    # toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) #todo: dodać metodę mutacji (nie flip bit)
    if config_dict["MUTATION"]["name"] == "mutGaussian":
        mu = config_dict["MUTATION"]["mu"]
        sigma = config_dict["MUTATION"]["sigma"]
        indpb = config_dict["MUTATION"]["indpb"]
        toolbox.register("mutate", tools.mutGaussian, mu=mu, sigma=sigma, indpb=indpb)
    else:
        raise NotImplementedError

    # wybór metody selekcji
    toolbox.register("select", tools.selTournament, tournsize=2)

    return toolbox


def prepare_statistics():
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    return stats


if __name__ == "__main__":
    try:
        config_filename = input("Wpisz nazwę pliku konfiguracyjnego lub pomiń aby użyć config.xml")
        if not exists(config_filename):
            input("Nie znaleziono podanego pliku konfiguracyjnego. Zostanie użyty config.xml")
            config_filename = "config.xml"

        parser = XmlParser(config_filename)
        config_dict = parser.get_config_dict()

        toolbox = prepare_toolbox(config_dict)
        stats = prepare_statistics()


        init_pop = toolbox.population(n=config_dict['POP_SIZE'])
        algorithm = config_dict["ALGORITHM"]

        rpop, logbook = None, None

        if algorithm == "eaSimple":
            rpop, logbook = algorithms.eaSimple(init_pop, toolbox, config_dict['CXPB'], config_dict['MUTPB'], config_dict['NGEN'], stats=stats)
        elif algorithm == "eaMuPlusLambda":
            mu = 20
            lambda_ = 7*mu
            rpop, logbook = algorithms.eaMuPlusLambda(init_pop, toolbox, mu, lambda_, config_dict['CXPB'], config_dict['MUTPB'], config_dict['NGEN'], stats=stats)
        # elif algorithm == "CustomSimple":
        #     main(config_dict, toolbox)
        else:
            print(f"algorytm o nazwie {algorithm} nie jest obsługiwany")

        min_ = logbook.select("min")
        plt.plot(min_)
        plt.show()

    except XmlParserError as xpe:
        print("wystąpił błąd podczas parsowania dokumentu konfiguracyjnego")
