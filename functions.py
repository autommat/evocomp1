import math

# definicje funkcji przystosowania do optymalizacji

# tylko do testów
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

def sin_times_cos_plus_sum(individual):
    x1 = individual[0]
    x2 = individual[1]
    y = math.sin(x1)*math.cos(x2)+x1+x2
    return y,