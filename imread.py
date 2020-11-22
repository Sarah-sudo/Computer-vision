import cv2
image = cv2.imread("C:/Users/sarah/Desktop/4.jpg")
print(image.shape)
cv2.imshow("image", image)
cv2.waitKey(0)