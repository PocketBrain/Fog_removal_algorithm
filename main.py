import cv2
from test_image import getRecoverScene
import filter
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.filedialog as fd
filename = fd.askopenfilename()
img = cv2.imread(filename)
image = cv2.resize(img, (860, 660))
dehazed_img = getRecoverScene(image, refine=False)
cv2.imshow('hazed image', image)
#cv2.waitKey(0)

cv2.imshow('final image', dehazed_img)
cv2.waitKey(0)
