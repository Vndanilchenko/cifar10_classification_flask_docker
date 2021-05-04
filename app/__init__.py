from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Flatten, Dense, Conv2D, Dropout, Input, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import cv2
import numpy as np



class Model_train:

    def __init__(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        self.y_train = to_categorical(y_train, num_classes=10)
        self.y_test = to_categorical(y_test, num_classes=10)

        # нормализуем данные
        x_train = x_train.astype('float')
        self.x_train = x_train / 255.
        x_test = x_test.astype('float')
        self.x_test = x_test / 255.


    def compile(self):
        inp = Input(shape=(32, 32, 3))
        # x = Conv2D(128, (9,9), padding='same', name='conv2_1', activation='relu')(inp)
        # x = MaxPooling2D((2, 2), strides=2)(x)
        x = Conv2D(32, (3, 3), padding='same', name='conv2_1', kernel_initializer='he_uniform', activation='relu')(inp)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), padding='same', name='conv2_2', kernel_initializer='he_uniform', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)
        x = Conv2D(64, (3, 3), padding='same', name='conv2_3', kernel_initializer='he_uniform', activation='relu')(inp)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), padding='same', name='conv2_4', kernel_initializer='he_uniform', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.3)(x)
        x = Conv2D(128, (3, 3), padding='same', name='conv2_5', kernel_initializer='he_uniform', activation='relu')(inp)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), padding='same', name='conv2_6', kernel_initializer='he_uniform', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.4)(x)
        x = Flatten(name='flatten')(x)
        x = Dense(128, activation='relu', kernel_initializer='he_uniform', name='dense_1')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        out = Dense(10, activation='softmax', name='dense_out')(x)

        model = Model(inp, out)
        model.summary()

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics='accuracy')

        model.load_weights('./cls/best_weights/model_weights.hdf5')
        self.model = model


    def train(self):
        callbacks = [ModelCheckpoint('./cls/model_weights.hdf5', monitor='val_accuracy', mode='max', save_weights_only=True, save_best_only=True), ]
                    # ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, mode='min', min_delta=0.001, min_lr=0.00001)]

        self.model.fit(self.x_train,
                  self.y_train,
                  batch_size=64,
                  epochs=50,
                  validation_data=(self.x_test, self.y_test),
                  verbose=1,
                  # shuffle=True,
                  callbacks=callbacks, )

    def predict(self, X):
        if X.shape == (32, 32, 3):
            X = X.reshape(1, 32, 32, 3)
        try:
            res = self.model.predict(X)
        except:
            self.compile()
            res = self.model.predict(X)
        return [int(i) for i in list(np.argmax(res, axis=1))]


if __name__ == '__main__':
    model = Model_train()
    model.compile()
    print('true:', np.argmax(model.y_test[0]),
          'pred:', model.predict(model.x_test[0]))
