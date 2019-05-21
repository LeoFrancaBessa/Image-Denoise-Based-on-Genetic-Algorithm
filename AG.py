import cv2
from skimage.measure import compare_ssim as ssim
from random import randint
import operator
import random
#Load Images
img_normal = None
img_noise = None

#Create First Population
def firstPopulation(size):
    population = []
    for i in range(size):
        population.append((randint(1, 40), randint(1, 40), randint(1, 10)))
    return population

#Fitness function
def fitness(h, twindows, swindows):
    
    img_denoise = cv2.fastNlMeansDenoising(img_noise,None, h, twindows, swindows)
    s_denoise = ssim(img_normal, img_denoise, multichannel=True)
    return s_denoise

#Determine the fitness for each individual
def computePerfPopulation(population):
	populationPerf = {}
	for i in population:
		populationPerf[i] = fitness(i[0], i[1], i[2])
	return sorted(populationPerf.items(), key = operator.itemgetter(1), reverse=True)

#Select the individuals for the crossover
def selectFromPopulation(populationSorted, best_sample, lucky_few):
	nextGeneration = []
	for i in range(best_sample):
		nextGeneration.append(populationSorted[i][0])
	for i in range(lucky_few):
		nextGeneration.append(random.choice(populationSorted)[0])
	random.shuffle(nextGeneration)
	return nextGeneration

#Create a child from 2 parents
def createChild(pai1, pai2):
    child = ((pai1[0], pai2[1], pai1[2]))
    return child

#Crossover
def next_generation(population):
    ng = population
    for i in range(0, (len(population)-1), 2):
        ng.append(createChild(population[i], population[i+2]))
        ng.append(createChild(population[i+2], population[i]))
    return ng


#Mutate for a single gene
def mutateGene(indi):
    aux = list(indi)
    index_modification = int(randint(0, len(indi)-2))
    if random.random() * 100 < 50:
        aux[index_modification] += randint(0, 5)
    else:
        aux[index_modification] -= randint(0, 5)
    return tuple(aux)

#Mutate population
def mutatePopulation(pop, chance_de_mut):
	for i in range(len(pop)):
		if random.random() * 100 < chance_de_mut:
			pop[i] = mutateGene(pop[i])
