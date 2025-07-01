from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50, DenseNet121
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from data.color_agu import apply_color_aug
import numpy as np

preprocess_map = {
    'MobileNetV2': mobilenet_preprocess,
    'EfficientNetB0': efficientnet_preprocess,
    'ResNet50': resnet_preprocess,
    'DenseNet121': densenet_preprocess,
}


def get_generators(model_name, input_shape=(224, 224, 3), batch_size=None, data_dir=None, augmentations=None):
    if data_dir is None:
        raise ValueError("❌ data_dir 값을 지정해야 합니다.")
    
    if augmentations is None:
        augmentations = []

    preprocess_func = preprocess_map[model_name]

    def custom_preprocessing(img):
        img = preprocess_func(img)
        img = apply_color_aug(img, augmentations)
        return img
    
    # folder 
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # train generator with augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )
    
    train_gen = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
    )

    # validation generator
    val_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )

    # test generator (no augmentation)
    test_gen = ImageDataGenerator(preprocessing_function=custom_preprocessing).flow_from_directory(
        test_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )

    return train_gen, val_gen, test_gen
