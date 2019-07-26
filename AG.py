import cv2
#from skimage.measure import compare_ssim as ssim
from astropy.stats import median_absolute_deviation as mad
from random import randint
import operator
import random

#Declarando as imagens como variaveis globais
#noise = ruído
img_noise = None


#Criando a primeira população, size será o numero de inviduos da população
def firstPopulation(size):
    population = []
    for i in range(size):
        #Inserindo um vetor de 3 posições com valores aleatorios na nossa população
        #Serão nossos parâmetros
        population.append((randint(1, 40), randint(1, 40), randint(1, 6)))
    return population


#Função que calcula o fitness de UM individuo, os valores passados são os parâmentros que queremos calcular
def fitness(h, twindows, swindows):
    #img_denoise será nossa imagem com o filtro aplicado
    img_denoise = cv2.fastNlMeansDenoising(img_noise,None, h, twindows, swindows)
    #Depois pegamos essa imagem com o filtro aplicado, e vemos seu nivel de ruido
    fit = 1/mad(img_denoise, axis=None)
    #fit será o valor do fitness de cada individuo
    return fit

#Função que calcula o fitness de TODA população
def computePerfPopulation(population):
	populationPerf = {}
	for i in population:
        #'i' irá receber cada invididuo da nossa população
        #Pegamos o valor dos 3 parâmetros de 'i' e jogamos na nossa função de fitness
		populationPerf[i] = fitness(i[0], i[1], i[2])
    #Um vetor organizado do maior para o menor fitness será retornado
	return sorted(populationPerf.items(), key = operator.itemgetter(1), reverse=True)


#Função para fazer a seleção de individuos da população
#best_sample = numero de individuos bons que queremos selecionar
#lucky_few = numero de individuos aleatorios que queremos pegar
def selectFromPopulation(populationSorted, best_sample, lucky_few):
	nextGeneration = []
    #Inserindo os melhores individuos na população dos pais
	for i in range(best_sample):
		nextGeneration.append(populationSorted[i][0])
    ##Inserindo individuos aleatorios na população dos pais
	for i in range(lucky_few):
		nextGeneration.append(random.choice(populationSorted)[0])
	random.shuffle(nextGeneration)
	return nextGeneration


#Função para fazer cromossomo de dois individuos e gerar o filho
def createChild(pai1, pai2):
    child = ((pai1[0], pai1[1], pai2[2]))
    #retorna um filho
    return child

#Função para pegar toda a população de pais de gerar filhos
def next_generation(population):
    ng = population
    #Percorre todo o nosso vetor de pais e faz o cruzamento
    for i in range(0, (len(population)-1), 2):
        ng.append(createChild(population[i], population[i+2]))
        ng.append(createChild(population[i+2], population[i]))
    #Retorna a proxima geração
    return ng


#Função para mutar o gene
#Recebe um individuo como parâmentro
def mutateGene(indi):
    aux = list(indi)
    #Escolhe aleatoriamente qual gene irá ser mutado
    index_modification = int(randint(0, len(indi)-2))
    #Chance de 50% do valor do gene ser subtraido ou somado
    if random.random() * 100 < 50:
        #Somando um valor aleatorio no gene
        aux[index_modification] += randint(0, 5)
    else:
        #Subtraindo um valor aleatorio do gene
        aux[index_modification] -= randint(0, 5)
    return tuple(aux)

#Função para fazer a mutação da população
#Recebe como parâmentro a chance de mutação
def mutatePopulation(pop, chance_de_mut):
	for i in range(len(pop)):
		if random.random() * 100 < chance_de_mut:
            #Se entrar no 'if' um individuo da nossa população será mutado
			pop[i] = mutateGene(pop[i])
