import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

resim_adresler = os.listdir("veriseti")

img = cv2.imread("veriseti/"+resim_adresler[0])
img = cv2.resize(img,(500,500))

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()

img_bgr = img
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ir_img = cv2.medianBlur(img_gray,5)
ir_img = cv2.medianBlur(img_gray,5)

plt.imshow(ir_img,cmap="gray")
plt.show()

medyan = np.median(ir_img)
low = 0.67*medyan
high = 1.33*medyan
kenarlik = cv2.Canny(ir_img,low,high)

plt.imshow(kenarlik,cmap="gray")
plt.show()

kenarlik = cv2.dilate(kenarlik,np.ones((3,3),np.uint8),iterations=1)

plt.imshow(kenarlik,cmap="gray")
plt.show()
