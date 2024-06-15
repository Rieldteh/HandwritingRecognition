import json
import random
from tqdm import tqdm
import numpy as np

from app.dataset.image.util import label_to_indexes

CONFIG = json.load(open('sample.json'))


class Indexer:
    def __init__(self, count, randomization=True):
        self.count = count
        self.index = 0
        self.indexes = list(range(self.count))
        self.randomization = randomization

    def get_index(self):
        if self.randomization and self.index == 0:
            random.shuffle(self.indexes)

        result = self.indexes[self.index]
        self.index = (self.index + 1) % self.count
        return result


class Sampler:
    def __init__(self, samples, process_text):
        self.samples = samples
        self.process_text = process_text
        self.count = len(samples)
        self.images = np.zeros((self.count, CONFIG["image"]["height"], CONFIG["image"]["width"]))
        self.labels = []
        self.indexer = Indexer(self.count)
        self.samples_created = False

    def __create_samples(self):
        word_counter = 0
        for word in tqdm(self.samples, desc=self.process_text):
            self.images[word_counter, :, :] = word.image
            self.labels.append(word.text)
            word_counter += 1

    def get_steps(self):
        return self.count // CONFIG["size"]

    def __create_inputs(self, parts):
        a, b, c, d = parts
        return {"the_input": a, "the_labels": b, "input_length": c, "label_length": d}

    def __fill_sample(self, parts):
        images, labels, input_length, label_length = parts
        for i in range(CONFIG["size"]):
            index = self.indexer.get_index()
            image, label = self.images[index], self.labels[index]
            images[i] = np.expand_dims(image.T, -1)
            labels[i, :len(label)] = label_to_indexes(CONFIG["labels"], label)
            label_length[i] = len(label)

    def __get_sample_parts(self):
        size = CONFIG["size"]
        images = np.ones([size, CONFIG["image"]["width"], CONFIG["image"]["height"], 1])
        labels = np.zeros([size, CONFIG["max_text_len"]])
        input_length = np.ones((size, 1)) * CONFIG["input_length"]
        label_length = np.zeros((size, 1))
        parts = images, labels, input_length, label_length
        self.__fill_sample(parts)
        return parts

    def next(self):
        if not self.samples_created:
            self.__create_samples()

        while True:
            parts = self.__get_sample_parts()
            inputs = self.__create_inputs(parts)
            outputs = {'ctc': np.zeros([CONFIG["size"]])}
            yield inputs, outputs
