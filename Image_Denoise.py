import AG
import matplotlib.image as mpimg
from datetime import datetime

start_time = datetime.now()

AG.img_normal = mpimg.imread("chest_normal.jpg", 1)
AG.img_noise = mpimg.imread("chest_noise.jpg", 1)

population = AG.firstPopulation(12)
fit_first_pop = AG.computePerfPopulation(population)

tracker = []

i=0
while(i<=10):
    sorted_population =  AG.computePerfPopulation(population)
    tracker.append(sorted_population[0][1])
    '''
    if(sorted_population[0][1] >= 0.8):
        break
    '''
    pais = AG.selectFromPopulation(sorted_population, 2, 4)
    population = AG.next_generation(pais)
    AG.mutatePopulation(population, 20)
    i+=1


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

