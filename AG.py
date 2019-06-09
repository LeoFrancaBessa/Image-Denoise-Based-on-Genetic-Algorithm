import cv2
#from skimage.measure import compare_ssim as ssim
from astropy.stats import median_absolute_deviation as mad
from random import randint
import operator
import random

img_normal = None
img_noise = None


def firstPopulation(size):
    population = []
    for i in range(size):
        population.append((randint(1, 40), randint(1, 40), randint(1, 6)))
    return population


def fitness(h, twindows, swindows):
    
    img_denoise = cv2.fastNlMeansDenoising(img_noise,None, h, twindows, swindows)
    s_denoise = 1/mad(img_denoise, axis=None)
    return s_denoise


def computePerfPopulation(population):
	populationPerf = {}
	for i in population:
		populationPerf[i] = fitness(i[0], i[1], i[2])
	return sorted(populationPerf.items(), key = operator.itemgetter(1), reverse=True)


def selectFromPopulation(populationSorted, best_sample, lucky_few):
	nextGeneration = []
	for i in range(best_sample):
		nextGeneration.append(populationSorted[i][0])
	for i in range(lucky_few):
		nextGeneration.append(random.choice(populationSorted)[0])
	random.shuffle(nextGeneration)
	return nextGeneration


def createChild(pai1, pai2):
    child = ((pai1[0], pai1[1], pai2[2]))
    return child

def next_generation(population):
    ng = population
    for i in range(0, (len(population)-1), 2):
        ng.append(createChild(population[i], population[i+2]))
        ng.append(createChild(population[i+2], population[i]))
    return ng


def mutateGene(indi):
    aux = list(indi)
    index_modification = int(randint(0, len(indi)-2))
    if random.random() * 100 < 50:
        aux[index_modification] += randint(0, 5)
    else:
        aux[index_modification] -= randint(0, 5)
    return tuple(aux)

def mutatePopulation(pop, chance_de_mut):
	for i in range(len(pop)):
		if random.random() * 100 < chance_de_mut:
			pop[i] = mutateGene(pop[i])