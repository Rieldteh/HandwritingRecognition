from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

from app.dataset.image.sampler import Sampler


class Fit:
    WEIGHTS_PATH = 'Models/Best_Model-{epoch:02d}-{val_loss:.3f}.h5'
    MONITOR = 'val_loss'
    EPOCHS = 6

    def __init__(self, model, train_data, validation_data):
        self.model = model
        self.train_data = train_data
        self.validation_data = validation_data

    def __get_checkpoint_callback(self):
        return ModelCheckpoint(
            filepath=self.WEIGHTS_PATH, monitor=self.MONITOR,
            verbose=1, save_best_only=True, save_weights_only=True
        )

    def __get_stop_callback(self):
        return EarlyStopping(
            monitor=self.MONITOR, min_delta=0, patience=10, verbose=0, mode='min'
        )

    def run(self):
        train_sampler = Sampler(self.train_data, "Сегментация тренировочных данных")
        validation_sampler = Sampler(self.validation_data, "Сегментация валидационных данных")
        self.model.fit_generator(generator=train_sampler.next(),
                                 steps_per_epoch=train_sampler.get_steps(),
                                 epochs=self.EPOCHS,
                                 validation_data=validation_sampler.next(),
                                 validation_steps=validation_sampler.get_steps(),
                                 callbacks=[self.__get_checkpoint_callback(), self.__get_stop_callback()])
