import h5py
from typing import Tuple, List
from tqdm import tqdm
from .word import Word


class IAMDataset:
    def __init__(self, dataset_path: str):
        self.__words = []
        self.__path = dataset_path

    def load(self):
        with h5py.File(self.__path, 'r') as hf:
            for key in tqdm(list(hf.keys()), desc="Обработка изображений"):
                group = hf[key]
                text = group['text'][()].decode('utf-8')
                image = group['image'][()]
                self.__words.append(Word(text, image))

    def __len__(self):
        return len(self.__words)

    def __getitem__(self, index):
        return self.__words[index]

    def get_train_and_validation_data(self, ratio=0.8) -> Tuple[List[Word], List[Word]]:
        split_index = int(ratio * len(self.__words))
        train_keys = self.__words[:split_index]
        validation_keys = self.__words[split_index:]
        return train_keys, validation_keys
