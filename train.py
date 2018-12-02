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
from keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout
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
BATCH_SIZE = 64
IMAGE_SIZE = (512, 512)
INCEPTIONV3_NO_TRAINABLE_LAYERS = 88
RESNET50_NO_TRAINABLE_LAYERS = 80
XCEPTION_NO_TRAINABLE_LAYERS = 86
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_PATIENCE = 3
CLASS_NUM = 2
DROP_RATE = 0.5


# -------------------------------------
# 文件路径
# -------------------------------------
TRAIN_DATA_PATH = './data/train/'
VALID_DATA_PATH = './data/valid/'
MODEL_PATH = './model/'


def finetuneModel(preprocess_input_func, base_model_class, model_name, no_trainable_layers):
    """
    preprocess_input_func: data preprocessing function
    base_model_class: model class
    model_name: model name
    no_trainable_layers: no trainable layers
    """
    train_gen = ImageDataGenerator()
    valid_gen = ImageDataGenerator()
    train_generator = train_gen.flow_from_directory(TRAIN_DATA_PATH, IMAGE_SIZE, shuffle=True, batch_size=BATCH_SIZE)
    valid_generator = valid_gen.flow_from_directory(VALID_DATA_PATH, IMAGE_SIZE, batch_size=BATCH_SIZE)

    inputs = Input((*IMAGE_SIZE, 3))
    x = Lambda(preprocess_input_func)(inputs)
    base_model = base_model_class(input_tensor=x, weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D(name='my_global_average_pooling_layer_1')(base_model.output)
    x = Dropout(DROP_RATE, name='my_dropout_layer_1')(x)
    predictions = Dense(CLASS_NUM, activation='softmax', name='my_dense_layer_1')(x)
    model = Model(base_model.input, predictions)
    plot_model(model, to_file='{}.png'.format(model_name), show_shapes=True)

    # set trainable layer
    for layer in model.layers[:no_trainable_layers]:
        layer.trainable = False
    for layer in model.layers[no_trainable_layers:]:
        layer.trainable = True

    # check
    layers = zip(range(len(model.layers)), [x.name for x in model.layers])
    for layer_num, layer_name in layers:
        print('{}: {}'.format(layer_num + 1, layer_name))

    # check point
    check_point = ModelCheckpoint(monitor='val_loss',
                                  filepath='./model/{}-finetune.hdf5'.format(model_name),
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='auto')

    # 早停
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, verbose=0, mode='auto')

    # 当评价指标不在提升时，减少学习率
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=REDUCE_LR_PATIENCE, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

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


def extractFeatures():

    filenames = ['xception-finetune.hdf5', 'resnet50-finetune.hdf5', 'inceptionv3-finetune.hdf5']

    train_gen = ImageDataGenerator()
    valid_gen = ImageDataGenerator()
    train_generator = train_gen.flow_from_directory(TRAIN_DATA_PATH, IMAGE_SIZE, shuffle=False, batch_size=BATCH_SIZE)
    valid_generator = valid_gen.flow_from_directory(VALID_DATA_PATH, IMAGE_SIZE, shuffle=False, batch_size=BATCH_SIZE)

    for filename in filenames:
        inputs = Input((*IMAGE_SIZE, 3))
        if filename == 'xception-finetune.hdf5':
            x = Lambda(xception.preprocess_input)(inputs)
            base_model = Xception(input_tensor=x, weights='imagenet', include_top=False)
        elif filename == 'resnet50-finetune.hdf5':
            x = Lambda(resnet50.preprocess_input)(inputs)
            base_model = ResNet50(input_tensor=x, weights='imagenet', include_top=False)
        elif filename == 'inceptionv3-finetune.hdf5':
            x = Lambda(inception_v3.preprocess_input)(inputs)
            base_model = InceptionV3(input_tensor=x, weights='imagenet', include_top=False)
        model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
        # 加载模型参数
        model.load_weights(os.path.join(MODEL_PATH, filename), by_name=True)
        train_features = model.predict_generator(train_generator, steps=len(train_generator.filenames) // BATCH_SIZE, use_multiprocessing=True, workers=8, verbose=1)
        valid_features = model.predict_generator(valid_generator, steps=len(valid_generator.filenames) // BATCH_SIZE, use_multiprocessing=True, workers=8, verbose=1)
        with h5py.File('{}-output.hdf5'.format(filename[:-5]), 'w') as h:
            h.create_dataset('X_train', data=train_features)
            h.create_dataset('y_train', data=train_generator.classes[:((train_generator.samples // BATCH_SIZE) * BATCH_SIZE)])
            h.create_dataset('X_val', data=valid_features)
            h.create_dataset('y_val', data=valid_generator.classes[:((valid_generator.samples // BATCH_SIZE) * BATCH_SIZE)])


def mergeFinetuneModel():

    X_train = []
    X_valid = []

    filenames = ['inceptionv3-finetune-output.hdf5', 'resnet50-finetune-output.hdf5', 'xception-finetune-output.hdf5']

    for filename in filenames:
        with h5py.File(filename, 'r') as h:
            X_train.append(np.array(h['X_train']))
            X_valid.append(np.array(h['X_val']))
            y_train = np.array(h['y_train'])
            y_valid = np.array(h['y_val'])

    X_train = np.concatenate(X_train, axis=1)
    X_valid = np.concatenate(X_valid, axis=1)

    # check
    print('X_train shape:', X_train.shape)
    print('X_valid shape:', X_valid.shape)
    print('y_train shape:', y_train.shape)
    print('y_valid shape:', y_valid.shape)

    X_train, y_train = shuffle(X_train, y_train)
    y_train = to_categorical(y_train)
    X_valid, y_valid = shuffle(X_valid, y_valid)
    y_valid = to_categorical(y_valid)

    print('X_train shape:', X_train.shape)
    print('X_valid shape:', X_valid.shape)
    print('y_train shape:', y_train.shape)
    print('y_valid shape:', y_valid.shape)

    inputs = Input(X_train.shape[1:])
    x = Dense(1024, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs, predictions)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    check_point = ModelCheckpoint(filepath='./model/weights.best.merge.hdf5',verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=1, mode='auto')
    callbacks_list = [early_stopping, check_point]
    history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_valid, y_valid), callbacks=callbacks_list)

if __name__ == '__main__':

    finetuneModel(inception_v3.preprocess_input, InceptionV3, 'inceptionv3', INCEPTIONV3_NO_TRAINABLE_LAYERS)
    #finetuneModel(resnet50.preprocess_input, ResNet50, 'resnet50', RESNET50_NO_TRAINABLE_LAYERS)
    #finetuneModel(xception.preprocess_input, Xception, 'xception', XCEPTION_NO_TRAINABLE_LAYERS)
    #extractFeatures()
    #mergeFinetuneModel()

