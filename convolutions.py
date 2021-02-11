# import necessary packages
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, K):
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]
    pad = (kW-1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")

    # sliding” the kernel from left-to-right and top-to-bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extracts the Region of Interest
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            # Convolution
            k = (roi * K).sum()
            output[y - pad, x - pad] = k
    
    # rescale output 
    output = rescale_intensity(output, in_range=(0,255))
    output = (output * 255).astype("uint8")
    return output

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# kernels
# for bluring
smallBlur = np.ones((7,7), dtype="float") * (1.0 / (7*7))
largeBlur = np.ones((21,21), dtype="float") * (1.0 / (21*21))

# for sharping
