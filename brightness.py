import cv2
import numpy as np

imagen = cv2.imread("/home/Pictures/sign_DoubleCurve.png")
imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
imagen = np.array(image1, dtype=np.float64)
random_bright = .5 + np.random.uniform()
image[:,:,2] = image[:,:,2] * random_bright
image[:,:,2][image[:,:,2]>255] = 255
image = np.array(image, dtype = np.uint8)
image = np.cv2.cvtColor(image)
cv2.show(image,"imagen.png")

