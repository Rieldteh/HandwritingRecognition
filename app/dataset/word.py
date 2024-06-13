import numpy as np


class Word:
    def __init__(self, text: str, image: np.ndarray):
        self.text = text
        self.image = image
