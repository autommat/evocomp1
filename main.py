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

POP_SIZE = 25
CXPB = 0.2
MUTPB = 0.5
NGEN = 50

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
        centroids = [individual[x:x + self.ncolumns] for x in range(0, len(individual), self.ncolumns)]
        clusters = self.individual_to_clusters(individual)

        sum_d_value = 0.

        # for i in range(NCLUSTERS):
        #     sum_d_value += self.calcDvalue(clusters, centroids)
        # davies_bouldin = sum_d_value/NCLUSTERS
        davies_bouldin = self.calcDvalue(clusters, centroids)

        return davies_bouldin,

    def individual_to_clusters(self, individual):
        clusters = {ncluster:[] for ncluster in range(NCLUSTERS)}
        centroids = [individual[x:x + self.ncolumns] for x in range(0, len(individual), self.ncolumns)]
        # print(centroids)

        for item_index, row in self.df.iterrows():
            point_coords = row.tolist()

            min_dist_index = -1
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
        return clusters

    def dist(self, center_coords:list, point_coords:list):
        dist = ((center_coords[0]-point_coords[0])**2 + (center_coords[1]-point_coords[1])**2)**0.5
        for cent_c, point_c in zip(center_coords[2:], point_coords[2:]):
            dist = (dist**2 + (cent_c-point_c)**2 )**0.5
        return dist

    def calcDvalue(self, clusters: map, centroids: list):
        s_cache = {x: None for x in clusters.keys()}
        maxR = 0
        for index1, cluster1 in enumerate(clusters):
            for index2, cluster2 in enumerate(clusters):
                if index1 != index2:
                    M = self.dist(centroids[index1], centroids[index2])
                    if s_cache[index1] is None:
                        s_cache[index1] = self.calcSvalue(centroids[index1], clusters[index1])
                    if s_cache[index2] is None:
                        s_cache[index2] = self.calcSvalue(centroids[index2], clusters[index2])
                    S1 = s_cache[index1]
                    S2 = s_cache[index2]
                    R = (S1+S2) / M
                    if R>maxR:
                        maxR = R
        return maxR

    def calcSvalue(self, centroid, cluster):
        if len(cluster) == 0:
            return 0.
        total_dist = 0.
        for index in cluster:
            point_coords = self.df.iloc[index]
            # point_coords = self.df.iloc[index].tolist()
            total_dist += self.dist(centroid, point_coords)
        return total_dist / len(cluster)

class RandInitializer:
    def __init__(self,nclusters, nattr, min=0., max=1.):
        self.nclusters= nclusters
        self.nattr = nattr
        self.min = min
        self.max = max
        self.update_list()
        self.current_rand = 0

    def update_list(self):
        l = [random.sample(range(self.nclusters), self.nclusters) for _ in range(self.nattr)]
        step = (self.max - self.min)/self.nclusters
        self.rands_list = []
        for cl in range(self.nclusters):
            for at in range(self.nattr):
                self.rands_list.append(step*l[at][cl] + step/2)

    def get_rand_val(self):
        self.current_rand+=1
        if self.current_rand>=len(self.rands_list):
            self.current_rand=0
            self.update_list()
        return self.rands_list[self.current_rand]


def prepare_toolbox(dataset: Dataset):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # typ i zakres genu osobnika
    # toolbox.register("attr_float", random.uniform, 0., 1.)
    rand_initialier = RandInitializer(NCLUSTERS, dataset.ncolumns, min=0., max=1.)
    toolbox.register("attr_float", rand_initialier.get_rand_val)
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
        for coord in individual:
            if coord<arg_min or coord > arg_max:
                return False
        return True

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
    toolbox.register("mate", tools.cxUniform, indpb=0.2)


    #wybór metody mutacji
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.3)

    # wybór metody selekcji
    toolbox.register("select", tools.selTournament, tournsize=2)

    return toolbox


def prepare_statistics():
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    return stats

if __name__ == "__main__":
    random.seed(1)
    dataset = Dataset()

    toolbox = prepare_toolbox(dataset)
    stats = prepare_statistics()


    init_pop = toolbox.population(POP_SIZE)

    elite = tools.HallOfFame(1)
    rpop, logbook = algorithms.eaSimple(init_pop, toolbox, CXPB, MUTPB, NGEN, stats=stats, halloffame=elite)
    best_clusters = dataset.individual_to_clusters(elite[0])
    best_indiv = elite[0]
    print(best_clusters)

    min_ = logbook.select("min")
    plt.plot(min_)
    plt.show()

    index_to_cluster = [None]*len(dataset.df.index)
    for clust_num, item_num_list in best_clusters.items():
        for item_num in item_num_list:
            index_to_cluster[item_num] = clust_num

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataset.df.iloc[:,0], dataset.df.iloc[:, 1], dataset.df.iloc[:,2], c=index_to_cluster)
    for clust in range(NCLUSTERS):
        i = clust*dataset.ncolumns
        ax.scatter(best_indiv[i+0], best_indiv[i+1], best_indiv[i+2], marker='^', c=0)
    plt.show()
    # plt.scatter(dataset.df.iloc[:, 2], dataset.df.iloc[:, 4], c=index_to_cluster)
    # plt.show()