import cv2
import numpy as np
from keras.models import load_model

from app.dataset.image.resizer import Resizer
from app.dataset.image.sampler import CONFIG
from app.dataset.image.segment import Segmentation
from app.dataset.image.util import decode
from app.model.fit import Fit


class Model:
    def __init__(self, mode):
        self.mode = mode

    def get_model(self):
        model_type = 'model_train.h5' if self.mode == 'train' else 'model_predict.h5'
        return load_model(model_type)

    def train(self, train_data, val_data):
        model = self.get_model()
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
        Fit(model, train_data, val_data).run()
        return model

    def predict(self, weights, image):
        model = self.get_model()
        model.load_weights(weights)
        image = cv2.imread(image)
        word_images = Segmentation(image).get_words()
        word_predictions = []
        for word_image in word_images:
            word_image = np.dstack([word_image, word_image, word_image])
            resizer = Resizer(word_image, CONFIG["image"]["width"], CONFIG["image"]["height"])
            processed_image = resizer.resize()
            expanded_image = np.expand_dims(processed_image.T, 0)
            output_parameters = model.predict(expanded_image)
            predict = decode(CONFIG["labels"], output_parameters)
            word_predictions.append(predict)

        sentence = ' '.join(word_predictions)
        print(f'Predict: {sentence}')
