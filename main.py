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
import pandas as pd


POP_SIZE = 20
CXPB = 0.2
MUTPB = 0.01
NGEN = 100
NCLUSTERS=3
NSAMPLES=210

class Dataset:
    def __init__(self):
        # self.df_raw = pd.read_csv('res/seeds_dataset.csv', header=None,
        #                  names=["area", "perimeter", "compactness", "length of kernel", "width of kernel",
        #                         "asymmetry coefficient", "length of kernel groove", "type"])
        # self.df = self.df_raw.drop(columns=['type'])
        self.df_raw = pd.read_csv('res/normalizedSeeds.csv', header=None,
                         names=["index","area", "perimeter", "compactness", "length of kernel", "width of kernel",
                                "asymmetry coefficient", "length of kernel groove"])
        self.df = self.df_raw.drop(columns=['index'])
        self.ncolumns = len(self.df.columns)

    def evaluate(self, individual):

        for x in individual:
            if x<0:
                print(x)


        clusters = {ncluster:[] for ncluster in range(NCLUSTERS)}
        # print(clusters)

        centroids = [individual[x:x+self.ncolumns] for x in range(0,len(individual), self.ncolumns)]
        # print(centroids)

        for item_index, row in self.df.iterrows():
            point_coords = row.tolist()

            min_dist_index=-1
            min_dist = None
            for centroid_index, centroid in enumerate(centroids):
                dist = self.dist(centroid, point_coords)
                if min_dist_index == -1:
                    min_dist_index = centroid_index
                    min_dist = dist
                else:
                    if min_dist > dist:
                        min_dist = dist
                        min_dist_index = centroid_index

            clusters[min_dist_index].append(item_index)
        # print(clusters)

        return -1,

    def dist(self, center_coords:list, point_coords:list):
        dist = ((center_coords[0]-point_coords[0])**2 + (center_coords[1]-point_coords[1])**2)**0.5
        for cent_c, point_c in zip(center_coords[2:], point_coords[2:]):
            dist = (dist**2 + (cent_c-point_c)**2 )**0.5
        return dist

def prepare_toolbox(dataset: Dataset):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # typ i zakres genu osobnika
    toolbox.register("attr_float", random.uniform, 0., 1.)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, NCLUSTERS*dataset.ncolumns)

    # toolbox.register("indices", random.sample, range(210), 210)
    # toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)

    # toolbox.register("attr_int", random.randint, 0, NSAMPLES)
    # toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_int, 4)


    # populacja to lista osobników
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # wybór funkcji przystosowania
    toolbox.register("evaluate", dataset.evaluate)

    # funkcja kary
    arg_min = 0
    arg_max = 1
    def feasible(individual):
        for each in individual:
            if arg_min < each < arg_max:
                return True
        return False

    def distance(individual):
        # tuple to float
        fit_val = dataset.evaluate(individual)[0]

        for x in individual:
            if x < arg_min:
                fit_val += 10 ** (1 + (arg_min - x))
            if x > arg_max:
                fit_val += 10 ** (1 + (x - arg_max))

        return fit_val,

    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 0.0, distance))

    # wybór metody krzyżowania
    # toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mate", tools.cxUniform, indpb=0.2)


    #wybór metody mutacji
    # indpb prawdopodobieństwo mutacji atrybutu/genu
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    # toolbox.register("mutate", tools.mutUniformInt, low=0, up=NSAMPLES, indpb=0.2)

    # wybór metody selekcji
    toolbox.register("select", tools.selTournament, tournsize=2)

    return toolbox


def prepare_statistics():
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    return stats


if __name__ == "__main__":
    # random.seed(2)
    dataset = Dataset()

    toolbox = prepare_toolbox(dataset)
    stats = prepare_statistics()


    init_pop = toolbox.population(POP_SIZE)


    rpop, logbook = algorithms.eaSimple(init_pop, toolbox, CXPB, MUTPB, NGEN, stats=stats)


    min_ = logbook.select("min")
    plt.plot(min_)
    plt.show()

