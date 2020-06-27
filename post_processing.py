import cv2
import numpy as np

image = cv2.imread("test_images/hinh-anh-nhung-chu-cho-de-thuong-nhat.jpg")
trimap = cv2.imread("test_images/result_2.png", cv2.IMREAD_GRAYSCALE)
# mask_image = np.any(image, axis=2)
# mask_trimap= np.any(trimap)
# mask_image[not mask_trimap] = 0
# print(mask_image.shape)

mask_trimap = trimap == 0
print(mask_trimap.shape)

new_image = np.zeros((image.shape[0], image.shape[1], 4))
new_image[:,:,0:3] = image

alpha_channel = np.full((image.shape[0], image.shape[1]), 255)
alpha_channel[mask_trimap] = 0 

new_image[:,:,3] = trimap

cv2.imwrite("test_images/transparent.png", new_image)

# # transparent = cv2.imread("test_images/transparent.png", cv2.IMREAD_UNCHANGED)
# # print(transparent.shape)