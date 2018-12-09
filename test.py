import cv2
import filters
import numpy as np

image = cv2.imread("img/cat.jpg")
# image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
portra = np.zeros(image.shape, dtype="uint8")
provia = np.zeros(image.shape, dtype="uint8")
velvia = np.zeros(image.shape, dtype="uint8")
cross_process = np.zeros(image.shape, dtype="uint8")

# filters.recolorCMV(image, image)
# cv2.imshow("cmv", image)
# cv2.waitKey(0)
# image = cv2.imread("img.jpeg")
# filters.recolorRC(image, image)
# cv2.imshow("cmv", image)
# cv2.waitKey(0)
# image = cv2.imread("img.jpeg")
# filters.recolorRGV(image, image)
# cv2.imshow("cmv", image)
# print(image[0])
# cv2.waitKey(0)
# image = cv2.imread("img.jpeg")
# print(image[0])
# cv2.imshow("cmv", image)
# cv2.waitKey(0)


# portra_filter = filters.BGRPortraCurveFilter()
# provia_filter = filters.BGRProviaCurveFilter()
# velvia_filter = filters.BGRVelviaCurveFilter()
# cross_filter = filters.BGRCrossProcessCurveFilter()

# portra_filter.apply(image, portra)
# provia_filter.apply(image, provia)
# velvia_filter.apply(image, velvia)
# cross_filter.apply(image, cross_process)

# images = np.hstack([image, portra, provia, velvia, cross_process])

filters.strokeEdges(image, image)
cv2.imshow("images", image)
cv2.waitKey(0)
