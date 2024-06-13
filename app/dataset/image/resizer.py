import cv2
import numpy as np


class Resizer:
    def __init__(self, image, width, height):
        self.__image = image
        self.__width = width
        self.__height = height

    def __expand(self, img, width, height):
        diff_height = self.__height - height
        diff_width = self.__width - width
        height_center = int(diff_height / 2)
        width_center = int(diff_width / 2)
        h1, h2 = height_center, height_center + height
        w1, w2 = width_center, width_center + width
        expanded = np.ones([self.__height, self.__width, 3]) * 255
        expanded[h1:h2, w1:w2, :] = img
        return expanded

    def calculate_first_condition(self, real_height, real_width):
        new_width = self.__width
        new_height = int(real_height * new_width / real_width)
        return new_width, new_height

    def calculate_second_condition(self, real_height, real_width):
        new_h = self.__height
        new_width = int(real_width * new_h / real_height)
        return new_width, new_h

    def calculate_third_condition(self, real_height, real_width):
        ratio = max(real_width / self.__width, real_height / self.__height)
        new_width = max(min(self.__width, int(real_width / ratio)), 1)
        new_height = max(min(self.__height, int(real_height / ratio)), 1)
        return new_width, new_height

    def __crop(self, img):
        real_height, real_width = img.shape[:2]
        res = [[None, Resizer.calculate_second_condition],
               [Resizer.calculate_first_condition, Resizer.calculate_third_condition]]
        calc_func = res[real_width >= self.__width][real_height >= self.__height]
        new_width, new_height = real_width, real_height

        if calc_func:
            new_width, new_height = calc_func(self, real_height, real_width)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return self.__expand(img, new_width, new_height)

    def __clean(self, img):
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(np.float32) / 255

    def resize(self):
        cropped = self.__crop(self.__image)
        return self.__clean(cropped)
