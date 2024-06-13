import env
import argparse
import os
from dataset.iam import IAMDataset
from model.model import Model


class HandwritingParser:
    MODEL_FORMAT_ERROR = 'Недопустимый формат датасета. Поддерживается только .h5'
    MODEL_NOT_FOUND_ERROR = 'Модель не найдена'
    DATASET_NOT_FOUND_ERROR = 'Датасет не найден'
    IMAGE_FORMAT_ERROR = 'Недопустимый формат изображения. Поддерживаются только .png и .jpg'
    IMAGE_NOT_FOUND_ERROR = 'Изображение не найдено'

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Распознование рукописного текста')
        self.parser.add_argument('-p', '--predict', action='store_true', help='Включить режим предсказания')
        self.parser.add_argument('-t', '--train', action='store_true', help='Включить режим обучения')
        self.parser.add_argument('-i', '--input_image', type=str, help='Путь к изображению')
        self.parser.add_argument('-m', '--model_path', type=str, help='Путь к модели')
        self.parser.add_argument('-d', '--dataset', type=str, help='Путь к датасету .h5')

    def parse(self):
        args = self.parser.parse_args()

        mode = 'predict' if args.predict else 'train' if args.train else ''
        args_dict = {
            "mode": mode,
            "image": args.input_image,
            "model": args.model_path,
            "dataset": args.dataset
        }

        self.__verify_args(args_dict)
        return args_dict

    def __verify_args(self, args):
        if args['mode'] == 'predict':
            self.__verify_predict_args(args)
        elif args['mode'] == 'train':
            self.__verify_train_args(args)
        else:
            raise ValueError("Необходимо выбрать режим: предсказание (-p) или обучение (-t).")

    def __verify_file_exists(self, path, message):
        if not path or not os.path.exists(path):
            raise ValueError(message)

    def __verify_file_format(self, file_path, formats, message):
        if not any(file_path.endswith(fmt) for fmt in formats):
            raise ValueError(message)

    def __verify_predict_args(self, args):
        self.__verify_file_exists(args['image'], self.IMAGE_NOT_FOUND_ERROR)
        self.__verify_file_format(args['image'], ['.png', '.jpg'], self.IMAGE_FORMAT_ERROR)
        self.__verify_file_exists(args['model'], self.MODEL_NOT_FOUND_ERROR)
        self.__verify_file_format(args['model'], ['.h5'], self.MODEL_FORMAT_ERROR)

    def __verify_train_args(self, args):
        self.__verify_file_exists(args['dataset'], self.DATASET_NOT_FOUND_ERROR)
        self.__verify_file_format(args['dataset'], ['.h5'], self.MODEL_FORMAT_ERROR)


class HandwritingRecognition:
    def __init__(self, _args):
        self.args = _args

    def main(self):
        model = Model(self.args['mode'])
        if self.args['mode'] == 'train':
            print("Вы выбрали режим обучения.")
            dataset = IAMDataset(self.args['dataset'])
            dataset.load()
            train_data, validation_data = dataset.get_train_and_validation_data()
            model.train(train_data, validation_data)
        else:
            print("Вы выбрали режим предсказания.")
            model.predict(self.args['model'], self.args['image'])


if __name__ == "__main__":
    handwriting_parser = HandwritingParser()
    #try:
    args = handwriting_parser.parse()
    HandwritingRecognition(args).main()
    #except ValueError as err:
        #print(str(err))
    #    handwriting_parser.parser.print_help()
