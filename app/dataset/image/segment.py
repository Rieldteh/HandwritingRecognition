import cv2
import numpy as np

from app.dataset.image.sampler import CONFIG

ANISOTROPIC_POWER = 5


def normalize_kernel(kernel):
    return kernel / np.sum(kernel)


def create_kernel(size):
    ker = np.zeros((size, size))
    return ker, size // 2


def craft_anisotropic_term(coordinate, sigma_one, sigma_two):
    numerator = coordinate ** 2 - sigma_one ** 2
    denominator = 2 * np.pi * (sigma_one ** ANISOTROPIC_POWER) * sigma_two
    return numerator / denominator


def anisotropic_gauss_kernel(s_x, s_y, size):
    ker, center = create_kernel(size)

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            to_exponent = -x ** 2 / (2 * s_x) - y ** 2 / (2 * s_y)
            anisotropic_sum = craft_anisotropic_term(x, s_x, s_y) + craft_anisotropic_term(y, s_y, s_x)
            ker[i, j] = anisotropic_sum * np.exp(to_exponent)

    return normalize_kernel(ker)


class Segmentation:
    MIN_AREA = 100

    def __init__(self, image):
        self.image = image

    def __resize(self):
        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        factor = CONFIG["image"]["height"] / grayscale.shape[0]
        return cv2.resize(src=grayscale, dsize=None, fx=factor, fy=factor)

    def __clean(self, image):
        ker = anisotropic_gauss_kernel(s_x=51, s_y=153, size=51)
        filter_border_type = cv2.BORDER_REPLICATE
        filtered = cv2.filter2D(image, -1, ker, borderType=filter_border_type)
        filtered = filtered.astype(np.uint8)
        threshold_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        th, thresh = cv2.threshold(filtered, 0, 255, threshold_type)
        return 255 - thresh

    def __get_contours(self, thresh):
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def get_words(self):
        resized_image = self.__resize()
        thresh = self.__clean(resized_image)
        contours = self.__get_contours(thresh)
        coordinates_and_images = dict()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cropped = resized_image[y: y + h, x: x + w]
            if cv2.contourArea(contour) > self.MIN_AREA:
                coordinates_and_images[x] = cropped

        return [image for x, image in sorted(coordinates_and_images.items())]
