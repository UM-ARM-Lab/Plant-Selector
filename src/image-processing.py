"""
image-processing.py

This python file turns an RGB image to a binary image. Applies Edge detection and skeletonization for weed extraction.

Created by: Miguel Munoz
Date: July 7th, 2022
"""
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


def adaptive_gaussian_thresholding(img):
    img = cv.imread(img, 0)
    img = cv.medianBlur(img, 5)
    ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                               cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY, 11, 2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def otsu_thresholding(img):
    img = cv.imread(img, 0)
    # global thresholding
    ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    # Otsu's thresholding
    ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img, (65, 65), 0)
    ret3, th3 = cv.threshold(blur, 200, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # plot all the images and their histograms
    images = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()


def canny_edge_detection(img):
    img = cv.imread(img, 0)
    edges = cv.Canny(img, 200, 400)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detection', fontsize=30), plt.xticks([]), plt.yticks([])
    plt.show()


def skeletonization(img, original):
    """================WORKS VERY GOOD================="""
    # img_or = cv.imread(img, 1)

    # Open Image

    img_or = img
    final_image = cv.imread(original, 1)

    # Turn it into HSV and apply K-Means

    hsv = cv.cvtColor(img_or, cv.COLOR_BGR2HSV)
    erosion_reshaped = hsv.reshape((-1, 3))
    erosion_reshaped = np.float32(erosion_reshaped)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1000, 0.01)
    k = 5
    _, labels, (centers) = cv.kmeans(erosion_reshaped, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)
    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(hsv.shape)

    # Only get the cluster that corresponds to the weed

    # disable only the cluster number 2 (turn the pixel into black)
    masked_image = np.copy(segmented_image)
    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to disable
    cluster = 3
    masked_image[labels != cluster] = [0, 0, 0]
    # convert back to original shape
    masked_image = masked_image.reshape(segmented_image.shape)

    # Apply morphological transformations to the weed to fill in gaps

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    erosion = cv.erode(masked_image, kernel, iterations=2)
    dilation = cv.dilate(erosion, kernel, iterations=3)
    closing = cv.morphologyEx(dilation, cv.MORPH_CLOSE, kernel)
    weed = cv.erode(closing, kernel, iterations=2)
    weed = cv.cvtColor(weed, cv.COLOR_RGB2GRAY)

    # Plot different images

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(img_or), plt.xticks([]), plt.yticks([])
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image), plt.xticks([]), plt.yticks([])
    plt.title("K-means")

    plt.figure(2)
    plt.subplot(1, 3, 1)
    plt.title("Masked Image")
    plt.imshow(masked_image), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2)
    plt.title("Closing Gaps")
    plt.imshow(closing), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 3)
    plt.imshow(weed, cmap='gray'), plt.xticks([]), plt.yticks([])
    plt.title("Final Weed")

    # #################SKELETONIZATION PART########################
    size = np.size(weed)
    skel = np.zeros(weed.shape, np.uint8)

    # img_blurred = cv.medianBlur(weed, 15)
    # weed = cv.adaptiveThreshold(weed, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                             cv.THRESH_BINARY, 3, 2)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    done = False

    while not done:
        eroded = cv.erode(weed, element)
        temp = cv.dilate(eroded, element)
        temp = cv.subtract(weed, temp)
        skel = cv.bitwise_or(skel, temp)
        weed = eroded.copy()

        zeros = size - cv.countNonZero(weed)
        if zeros == size:
            done = True
    plt.figure(3)
    plt.imshow(skel, cmap='gray')
    plt.title('Skeletonization'), plt.xticks([]), plt.yticks([])

    # Final image is a copy of the original image. Here I "draw" the skeleton on top of the original image
    skel = cv.dilate(skel, kernel)

    for x in range(skel.shape[0]):
        for y in range(skel.shape[1]):
            if skel[x, y] != 0:
                final_image[x, y, 0] = 255
                final_image[x, y, 1] = 0
                final_image[x, y, 2] = 0

    # Display the final result: original image and skeleton of weed

    plt.figure(4)
    plt.imshow(final_image)
    plt.title('Skeleton'), plt.xticks([]), plt.yticks([])

    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()


def contours_canny(img):
    """==========WORKS==========="""
    # Let's load a simple img with 3 black squares
    img = cv.imread(img)
    cv.waitKey(0)

    # Grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Find Canny edges
    edged = cv.Canny(gray, 500, 700)
    cv.waitKey(0)

    # Finding Contours
    # Use a copy of the img e.g. edged.copy()
    # since findContours alters the img
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    plt.figure(1)
    plt.imshow(edged, cmap='gray')
    plt.title('Canny Edges After Contouring'), plt.xticks([]), plt.yticks([])
    cv.waitKey(0)

    print("Number of Contours found = " + str(len(contours)))

    # Draw all contours
    # -1 signifies drawing all contours
    cv.drawContours(img, contours, -1, (0, 255, 0), 3)

    plt.figure(2)
    plt.imshow(img, cmap='gray')
    plt.title('Contours'), plt.xticks([]), plt.yticks([])
    cv.waitKey(0)
    plt.show()
    cv.destroyAllWindows()


def remove_noise(img):
    image = cv.imread(img, 1)

    image_bw = cv.imread(img, 0)
    strength = 20
    noiseless_image_bw = cv.fastNlMeansDenoising(image_bw, None, strength, 7, 21)

    noiseless_image_colored = cv.fastNlMeansDenoisingColored(image, None, 30, strength, 7, 21)

    # titles = ['Original Image(colored)', 'Image after removing the noise (colored)', 'Original Image (grayscale)',
    #           'Image after removing the noise (grayscale)']
    # images = [image, noiseless_image_colored, image_bw, noiseless_image_bw]
    # plt.figure(figsize=(13, 5))
    # for i in range(4):
    #     plt.subplot(2, 2, i + 1)
    #     plt.imshow(cv.cvtColor(images[i], cv.COLOR_BGR2RGB))
    #     plt.title(titles[i])
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.tight_layout()
    # plt.show()
    return noiseless_image_colored


def main():
    img = '/home/miguel/Downloads/weed-alone.jpg'
    # adaptive_gaussian_thresholding(img)
    # otsu_thresholding(img)
    # canny_edge_detection(img)
    # skeletonization(img)
    # contours_canny(img)
    img_without_noise = remove_noise(img)
    skeletonization(img_without_noise, img)


if __name__ == '__main__':
    main()
