import AG
import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
import cv2


#Carrega Imagens
image = "normal2"
AG.img_noise = mpimg.imread(image+".jpeg", 1)


#Variaveis Globais
population = AG.firstPopulation(12)
fit_first_pop = AG.computePerfPopulation(population)


#Faz o trackeamento de fitness de cada geração
tracker = []


#Loop do AG
i=0
while(i<=30):
    sorted_population =  AG.computePerfPopulation(population)
    tracker.append(sorted_population[0][1])
    pais = AG.selectFromPopulation(sorted_population, 2, 4)
    population = AG.next_generation(pais)
    AG.mutatePopulation(population, 20)
    print(i)
    i+=1


#Pega o melhor individuo da população
best = sorted_population[0][0]


#Aplica o filtro a este individuo
img_denoise = cv2.fastNlMeansDenoising(AG.img_noise,None, best[0], best[1], best[2])


cv2.imshow("teste", img_denoise)
cv2.imwrite(image+'_filtro.jpeg', img_denoise)

'''
#Plotando a original
fig = plt.figure(num='Results')
ax1 = fig.add_subplot(2,2,1)
ax1.set_title("Original")
ax1.imshow(AG.img_noise)
plt.axis('off')


#Plotando a imagem com o filtro
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(AG.img_denoise)
plt.axis('off')


plt.figure(num='Fitness History')
plt.plot(tracker)
plt.ylabel('Fitness')
plt.xlabel('Nº de Gerações')
plt.show()
'''
