import cv2
from skimage.measure import compare_ssim as ssim
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img_normal = mpimg.imread("chest_normal.jpg", 1)
img_noise = mpimg.imread("chest_noise.jpg", 1)

#Plot Test
img_denoise = cv2.fastNlMeansDenoisingColored(img_noise,None, 30, 24, 18, 5)#0.7639196542961436


aux = ssim(img_normal, img_denoise5, multichannel=True)
aux2 = ssim(img_normal, img_noise, multichannel=True)

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.set_title("Original")
ax1.imshow(img_normal)
plt.axis('off')


ax2 = fig.add_subplot(2,2,2)
ax2.set_title("SSMI Filtro: " + str(aux))
ax2.imshow(img_denoise5_t)
plt.axis('off')


ax3 = fig.add_subplot(2,2,3)
ax3.set_title("SSMI Ruido: " + str(aux2))
ax3.imshow(img_noise)
plt.axis('off')


plt.show()
