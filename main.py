import cv2  # openCV python package
import numpy as np  # numpy package

print("Hello OpenCV")
shape = (640, 480)
image_matrix = np.ones_like(shape)
filepath = "Test Image.jpg"
image_create = cv2.imwrite(filepath, image_matrix)
image = cv2.imread(filepath)
cv2.imshow("Test  Image", image)
cv2.waitKey(0)