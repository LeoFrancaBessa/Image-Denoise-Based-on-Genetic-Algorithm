import AG
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
import cv2


#Load Images
AG.img_normal = mpimg.imread("chest_normal.jpg", 1)
AG.img_noise = mpimg.imread("chest_noise.jpg", 1)


#Global Variables
population = AG.firstPopulation(12)
fit_first_pop = AG.computePerfPopulation(population)


#Track the best individual from the population
tracker = []


#Genetic Algorithm flow with n interations(can change)
i=0
while(i<=20):
    sorted_population =  AG.computePerfPopulation(population)
    tracker.append(sorted_population[0][1])
    pais = AG.selectFromPopulation(sorted_population, 2, 4)
    population = AG.next_generation(pais)
    AG.mutatePopulation(population, 20)
    i+=1


#Get the best individual from the population
best = sorted_population[0][0]
img_denoise = cv2.fastNlMeansDenoising(AG.img_noise,None, best[0], best[1], best[2])


#Estimate the fitness of the solution to show in the plot
ssim_denoise = ssim(AG.img_normal, img_denoise, multichannel=True)
ssim_noise = ssim(AG.img_normal, AG.img_noise, multichannel=True)


#Ploting
fig = plt.figure(num='Results')
ax1 = fig.add_subplot(2,2,1)
ax1.set_title("Original")
ax1.imshow(AG.img_normal)
plt.axis('off')


ax2 = fig.add_subplot(2,2,2)
ax2.set_title("SSIM Filter: " + str(ssim_denoise))
ax2.imshow(img_denoise)
plt.axis('off')


ax3 = fig.add_subplot(2,2,3)
ax3.set_title("SSIM Noise: " + str(ssim_noise))
ax3.imshow(AG.img_noise)
plt.axis('off')


plt.figure(num='Fitness History')
plt.plot(tracker)
plt.ylabel('Fitness')
plt.xlabel('Nº de Gerações')
plt.show()