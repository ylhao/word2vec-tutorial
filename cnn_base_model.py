# coding: utf-8


import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import resnet50, inception_v3, xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.layers.core import Lambda
from keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout, Conv2D, MaxPooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import adam, Adam
from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model


# -------------------------------------
# 定义参数
# -------------------------------------
EPOCHS = 10
BATCH_SIZE = 32
IMAGE_SIZE = (512, 512)
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_PATIENCE = 3
CLASS_NUM = 2
DROP_RATE = 0.25


# -------------------------------------
# 文件路径
# -------------------------------------
TRAIN_DATA_PATH = './data/train/'
VALID_DATA_PATH = './data/valid/'
MODEL_PATH = './model/'


def create_cnn_base_model():
    
    train_gen = ImageDataGenerator()
    valid_gen = ImageDataGenerator()
    train_generator = train_gen.flow_from_directory(TRAIN_DATA_PATH, IMAGE_SIZE, shuffle=True, batch_size=BATCH_SIZE)
    valid_generator = valid_gen.flow_from_directory(VALID_DATA_PATH, IMAGE_SIZE, batch_size=BATCH_SIZE)

    inputs = Input((*IMAGE_SIZE, 3))

    # x = Conv2D(16, (3, 3), input_shape=(*IMAGE_SIZE, 3), activation='relu', name='Conv2D_1')(inputs)
    # x = MaxPooling2D(pool_size=(2, 2), name='MaxPooling2D_1')(x)
    # x = Dropout(DROP_RATE, name='Dropout_1')(x)
    
    x = Conv2D(32, (3, 3), input_shape=(*IMAGE_SIZE, 3), activation='relu', name='Conv2D_1')(inputs)
    x = MaxPooling2D(pool_size=(2, 2), name='MaxPooling2D_1')(x)
    x = Dropout(DROP_RATE, name='Dropout_1')(x)
    
    # x = Conv2D(64, (3, 3), input_shape=(*IMAGE_SIZE, 3), activation='relu', name='Conv2D_3')(x)
    # x = MaxPooling2D(pool_size=(2, 2), name='MaxPooling2D_3')(x)
    # x = Dropout(DROP_RATE, name='Dropout_3')(x)

    # x = Conv2D(128, (3, 3), input_shape=(*IMAGE_SIZE, 3), activation='relu', name='Conv2D_4')(x)
    # x = MaxPooling2D(pool_size=(2, 2), name='MaxPooling2D_4')(x)
    # x = Dropout(DROP_RATE, name='Dropout_4')(x)

    x = Conv2D(256, (3, 3), input_shape=(*IMAGE_SIZE, 3), activation='relu', name='Conv2D_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MaxPooling2D_2')(x)
    x = Dropout(DROP_RATE, name='Dropout_2')(x)

    x = GlobalAveragePooling2D(name='GlobalAveragePooling2D_1')(x)
    x = Dropout(DROP_RATE, name='Dropout_3')(x)

    predictions = Dense(CLASS_NUM, activation='softmax', name='Dense_1')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    model.summary()
    plot_model(model, to_file='cnn_base_model.png', show_shapes=True)

    # check point
    check_point = ModelCheckpoint(monitor='val_loss',
                                  filepath='./model/cnn_base_model.hdf5',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='auto')

    # early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, verbose=0, mode='auto')

    # reduce lr
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=REDUCE_LR_PATIENCE, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    # compile
    model.compile(optimizer=adam(lr=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

    # fit
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // BATCH_SIZE,
        callbacks=[check_point, early_stopping]
    )


if __name__ == '__main__':

    create_cnn_base_model()
